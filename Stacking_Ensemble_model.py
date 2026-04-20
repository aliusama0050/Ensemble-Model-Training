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
train_dataset_stacking = CustomImageDataset(x_train, y_train, transform=image_transform3)
test_dataset_stacking = CustomImageDataset(x_test, y_test, transform=image_transform3)
val_dataset_stacking = CustomImageDataset(x_val, y_val, transform=image_transform3)

train_loader_stacking = DataLoader(train_dataset_stacking, batch_size=16, shuffle=True)
test_loader_stacking = DataLoader(test_dataset_stacking, batch_size=16, shuffle=False)
val_loader_stacking = DataLoader(val_dataset_stacking, batch_size=16, shuffle=False)

# Stacing Ensembke model
class StackingMetaModel(pl.LightningModule):
    def __init__(self, resnet_model = ResNet_50_model_saved, inception_model = model_inception_v3_saved, alexnet_model = model_alexnet_saved, num_classes=9, y_train=y_train):
        super(StackingMetaModel, self).__init__()
        self.resnet_model = resnet_model
        self.inception_model = inception_model
        self.alexnet_model = alexnet_model

        self.meta_classifier = nn.Linear(3 * num_classes, num_classes)

        self.val_losses = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        if y_train is not None:
            y_train = torch.tensor(y_train)
            class_sample_count = torch.tensor([(y_train == t).sum() for t in torch.unique(y_train)])
            weight = 1. / class_sample_count.float()
            self.class_weights = torch.FloatTensor(weight).to('cuda')
        else:
            self.class_weights = None
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    def forward(self, x):
        resnet_output = self.resnet_model(x)
        inception_output = self.inception_model(x)
        alexnet_output = self.alexnet_model(x)

        if hasattr(inception_output, 'logits'):
            inception_output = inception_output.logits

        stacked_output = torch.cat((resnet_output, inception_output, alexnet_output), dim=1)
        return self.meta_classifier(stacked_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss)
        self.training_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
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
stacking_meta_model = StackingMetaModel()
callbacks = stacking_meta_model.configure_callbacks()
stacking_trainer = pl.Trainer(callbacks=callbacks, max_epochs=2)
stacking_trainer.gpus =2

stacking_trainer.fit(stacking_meta_model, train_loader_stacking, val_loader_stacking)

# Saving the model
path=''
torch.save(stacking_meta_model.state_dict(), path)

# Evaluate the model
stacking_meta_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stacking_meta_model.to(device)
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader_stacking:
        inputs = inputs.to(device)  
        targets = targets.to(device)  
        outputs = stacking_meta_model(inputs)
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