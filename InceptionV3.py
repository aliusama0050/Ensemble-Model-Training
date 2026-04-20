from lib_imports import pl,models,nn,np,torch,optim,transforms,DataLoader,precision_score,recall_score,f1_score,confusion_matrix
from dtasets import y_train, y_test, x_train, x_test, CustomImageDataset

# Loading the model
class InceptionV3Classifier(pl.LightningModule):
    def __init__(self, learning_rate=0.01, num_of_classes=9, y_train=None):
        super(InceptionV3Classifier, self).__init__()
        self.model = models.inception_v3(pretrained=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_of_classes)
        
        num_aux_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_of_classes)
        
        self.learning_rate = learning_rate
        
        if y_train is not None:
            class_sample_count = np.array([sum(y_train == t) for t in np.unique(y_train)])
            weight = 1. / class_sample_count
            self.class_weights = torch.FloatTensor(weight).to(self.device)
        else:
            self.class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self(inputs)

        main_output, aux_output = outputs.logits, outputs.aux_logits

        loss1 = self.criterion(main_output, labels)
        loss2 = self.criterion(aux_output, labels)
        loss = loss1 + 0.4 * loss2

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

def image_transform1(image):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Loading the training and testing data
train_dataset_InceptionV3 = CustomImageDataset(x_train, y_train, transform=image_transform1)
test_dataset_InceptionV3 = CustomImageDataset(x_test, y_test, transform=image_transform1)

train_loader_InceptionV3 = DataLoader(train_dataset_InceptionV3, batch_size=16, shuffle=True)
test_loader_InceptionV3 = DataLoader(test_dataset_InceptionV3, batch_size=16, shuffle=False)

#Checking the model architecture
inception_model = InceptionV3Classifier(y_train=y_train)
print(inception_model)

# Training the model

trainer = pl.Trainer(max_epochs=3)
trainer.gpus = 2

trainer.fit(inception_model, train_loader_InceptionV3)

#Saving the model
path=''
torch.save(inception_model.state_dict(), path)


# Testing the model
inception_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inception_model.to(device)
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader_InceptionV3:
        inputs = inputs.to(device) 
        targets = targets.to(device)  
        outputs = inception_model(inputs)
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

