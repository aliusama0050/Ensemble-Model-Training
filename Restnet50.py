from lib_imports import pl,resnet50,nn,np,torch,optim,transforms,DataLoader,precision_score,recall_score,f1_score,confusion_matrix
from dtasets import y_train, y_test, x_train, x_test, CustomImageDataset

# Loading the model
class ResNetClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.01, num_of_classes=9, y_train=y_train):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_of_classes)
        self.learning_rate = learning_rate

        class_sample_count = np.array([sum(y_train == t) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        self.class_weights = torch.FloatTensor(weight).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

def image_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Loading the training and testing data
train_dataset_Restnet_50 = CustomImageDataset(x_train, y_train, transform=image_transform)
test_dataset_Restnet_50 = CustomImageDataset(x_test, y_test, transform=image_transform)

train_loader_Restnet_50 = DataLoader(train_dataset_Restnet_50, batch_size=32, shuffle=True)
test_loader_Restnet_50 = DataLoader(test_dataset_Restnet_50, batch_size=32, shuffle=False)

#Checking the model architecture
RestNet_50_model = ResNetClassifier()
print(RestNet_50_model)

# Training the model
trainer = pl.Trainer(max_epochs=1)
trainer.gpus = 2

trainer.fit(RestNet_50_model, train_loader_Restnet_50)

# Saving the model
path=''
torch.save(RestNet_50_model.state_dict(), path)

# Evaluating the model
RestNet_50_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RestNet_50_model.to(device)
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader_Restnet_50:
        inputs = inputs.to(device)  
        targets = targets.to(device)  
        outputs = RestNet_50_model(inputs)
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