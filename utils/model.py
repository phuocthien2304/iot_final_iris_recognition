# model.py
import torch.nn as nn

class SimpleIrisCNN(nn.Module):
    def __init__(self,num):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.c=nn.Sequential(nn.Flatten(),nn.Dropout(0.5),nn.Linear(256*4*4,512),
                             nn.ReLU(),nn.Dropout(0.5),nn.Linear(512,num))
    def forward_features(self,x):
        x = self.f(x)
        x = self.c[0](x)
        x = self.c[1](x)
        x = self.c[2](x)
        x = self.c[3](x)
        return x
    def forward(self,x):return self.c(self.f(x))

class ImprovedIrisCNN(nn.Module):
    def __init__(self,num):
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.c=nn.Sequential(nn.Flatten(),nn.Dropout(0.4),nn.Linear(256*4*4,512),
                             nn.ReLU(),nn.Dropout(0.4),nn.Linear(512,num))
    def forward_features(self,x):
        x = self.f(x)
        x = self.c[0](x)
        x = self.c[1](x)
        x = self.c[2](x)
        x = self.c[3](x)
        return x
    def forward(self,x):return self.c(self.f(x))
