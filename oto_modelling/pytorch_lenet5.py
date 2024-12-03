import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd


class LeNet5(nn.Module):
    
    def __init__(self):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400,120),  #in_features = 16 x5x5 
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
            nn.Softmax()

        )
        
    def forward(self,x): 
        a1=self.feature_extractor(x)
        print(a1.shape)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2