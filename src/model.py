## UNDER COURSE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from timm import create_model
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision.transforms import v2

from dataLoader import dataloaderRegression

BATCH_SIZE = 32

class CustomFCN(nn.Module):
    """
    Custom FCN inspired by AngReg Topology without Tanh for better speed
    """
    def __init__(self):
        super(CustomFCN, self).__init__()
        self.fc1 = nn.Linear(2048, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        # self.dropout3 = nn.Dropout(0.5)
        # self.tanh = nn.Tanh()
        # self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.dropout3(x)
        # x = self.tanh(x)
        # x = self.fc4(x)

        return x
    
if __name__ == '__main__':

    seed = 7
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    labels_df = pd.read_csv(LABELS_FILE)
    labels_df.head()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    xception = create_model('xception', pretrained=True)

    # freeze early layers
    for param in xception.parameters():
        if param.requires_grad:
            param.requires_grad = False

    fcn = CustomFCN()
    # modify head
    xception.fc = fcn
    # Instantiate the custom FCN
    
    
    transform_validation = v2.Compose([
        v2.Resize(299),
        v2.CenterCrop(299),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define data augmentation transformations
    transform_train = v2.Compose([
        # torchvision.transforms.RandomResizedCrop(224),
        v2.Resize(299),
        v2.CenterCrop(299),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # standard stats for ImageNet
    ])

    train_loader = dataloaderRegression(train_dataset, TRAIN_DIR, transform_train, batch_size=BATCH_SIZE, isTrain=True) # ease training
    val_loader = dataloaderRegression(, TRAIN_DIR, transform_validation, batch_size=2*BATCH_SIZE, isTrain=False)


