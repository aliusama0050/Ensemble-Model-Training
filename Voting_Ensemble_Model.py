from lib_imports import pl,nn,np,torch,optim,transforms,DataLoader,precision_score,recall_score,f1_score,confusion_matrix,EarlyStopping
from dtasets import y_train, y_test, x_train, x_test, x_val, y_val, CustomImageDataset
from Alexnet import AlexNet
from InceptionV3 import InceptionV3Classifier
from Restnet50 import ResNetClassifier

# Enter the path to where the previous models which were trained on this dataset were saved
saved_model_path_resnet = ''
saved_model_path_inception = ''
saved_model_path_alexnet = ''

# Load the base models
ResNet_50_model_saved = ResNetClassifier(y_train=y_train)
ResNet_50_model_saved.load_state_dict(torch.load(saved_model_path_resnet))
ResNet_50_model_saved.eval()

model_inception_v3_saved = InceptionV3Classifier(y_train=y_train)
model_inception_v3_saved.load_state_dict(torch.load(saved_model_path_inception))
ResNet_50_model_saved.eval()

model_alexnet_saved = AlexNet(y_train=y_train)
model_alexnet_saved.load_state_dict(torch.load(saved_model_path_alexnet))
ResNet_50_model_saved.eval()

def image_transform3(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

# Load the training and testing and  validation data
train_dataset_voing = CustomImageDataset(x_train, y_train, transform=image_transform3)
test_dataset_voing = CustomImageDataset(x_test, y_test, transform=image_transform3)
val_dataset_voing = CustomImageDataset(x_val, y_val, transform=image_transform3)

train_loader_voing = DataLoader(train_dataset_voing, batch_size=32, shuffle=True)
test_loader_voing = DataLoader(test_dataset_voing, batch_size=32, shuffle=False)
val_loader_voing = DataLoader(val_dataset_voing, batch_size=32, shuffle=False)

# Stacing Ensembke model
class VotingMetaModel(pl.LightningModule):
    def __init__(self, resnet_model=ResNet_50_model_saved, inception_model=model_inception_v3_saved, alexnet_model=model_alexnet_saved, num_classes=9):
        super(VotingMetaModel, self).__init__()
        self.resnet_model = resnet_model
        self.inception_model = inception_model
        self.alexnet_model = alexnet_model

        self.val_losses = []
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    def forward(self, x):
        resnet_output = self.resnet_model(x)
        inception_output = self.inception_model(x)
        alexnet_output = self.alexnet_model(x)

        if hasattr(inception_output, 'logits'):
            inception_output = inception_output.logits

        outputs = [resnet_output, inception_output, alexnet_output]

        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log('train_loss', loss)
        self.training_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = nn.functional.cross_entropy(outputs, y)
        self.log('val_loss', loss)
        self.validation_step_outputs.append(loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log('avg_train_loss', avg_loss)
        self.training_step_outputs.clear()

    def configure_callbacks(self):
        return [self.early_stopping]
    
# Training the model
voing_meta_model_soft = VotingMetaModel()
callbacks = voing_meta_model_soft.configure_callbacks()
trainer_soft = pl.Trainer(max_epochs=3, callbacks=callbacks)
trainer_soft.gpus = 2

trainer_soft.fit(voing_meta_model_soft, train_loader_voing, val_loader_voing)


# Evaluate the model
voing_meta_model_soft.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voing_meta_model_soft.to(device)
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader_voing:
        inputs = inputs.to(device)  
        targets = targets.to(device)  
        outputs = voing_meta_model_soft(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(targets.cpu().numpy())

accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%')

precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1 Score: {:.4f}'.format(f1))

conf_matrix = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)