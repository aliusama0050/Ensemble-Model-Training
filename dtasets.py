#The dataset is from a pandas dataframe that has already been cleaned of any missing values. It is converted to lists and stored in following variables

from lib_imports import Dataset,Image

x_train = training_data["path"].tolist() #type: ignore
y_train = training_data["Label"].tolist() #type: ignore
x_val = validation_data["path"].tolist() #type: ignore
y_val = validation_data["Label"].tolist() #type: ignore
x_test = testing_data["path"].tolist() #type: ignore
y_test = testing_data["Label"].tolist() #type: ignore

#Custom class for loading the dataset

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target