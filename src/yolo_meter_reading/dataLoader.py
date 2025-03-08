import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image


# Custom Dataset Classes
class ImageRegressionDataset(Dataset):
    """
    Custom Dataset class for regression (training, and validation)
    Attributes
    annotations: DataFrame containing the image paths and index labels
    root_dir: the path to images
    transform: PyTorch to apply to dataset
    """
    def __init__(self, df, root_dir, transform=None):
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label
    

class ImageFileDataset(Dataset):
    """
    Dataset class without labels but paths of the image files
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)
    
    def dataloaderRegression(labels_df: pd.DataFrame, root_dir, transform, batch_size, isTrain=False) -> torch.utils.data.DataLoader:
        """
        Dataloader for regression task
        """
        dataset = ImageRegressionDataset(labels_df, root_dir, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=isTrain)
