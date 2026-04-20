from lib_imports import pl,nn,np,torch,optim,transforms,DataLoader,precision_score,recall_score,f1_score,confusion_matrix
from dtasets import y_train, y_test, x_train, x_test, CustomImageDataset

class AlexNet(pl.LightningModule):
    def __init__(self, num_classes=9, y_train=y_train):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        if y_train is not None:
            y_train = torch.tensor(y_train)
            class_sample_count = torch.tensor([(y_train == t).sum() for t in torch.unique(y_train)])
            weight = 1. / class_sample_count.float()
            self.class_weights = torch.FloatTensor(weight).to('cuda')
        else:
            self.class_weights = None
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.001)
    
    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

def image_transform2(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

# Loading the training and testing data
train_dataset_AlexNet = CustomImageDataset(x_train, y_train, transform=image_transform2)
test_dataset_AlexNet = CustomImageDataset(x_test, y_test, transform=image_transform2)

train_loader_AlexNet = DataLoader(train_dataset_AlexNet, batch_size=16, shuffle=True)
test_loader_AlexNet = DataLoader(test_dataset_AlexNet, batch_size=16, shuffle=False)

#Checking the model architecture
AlexNet_model = AlexNet()
print(AlexNet_model)

# Training the model
trainer = pl.Trainer(max_epochs=10)
trainer.gpus = 2

trainer.fit(AlexNet_model, train_loader_AlexNet)

# Saving the model
path=''
torch.save(AlexNet_model.state_dict(), path)

# Evaluating the model
AlexNet_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AlexNet_model.to(device)
correct = 0
total = 0
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader_AlexNet:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = AlexNet_model(inputs)
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