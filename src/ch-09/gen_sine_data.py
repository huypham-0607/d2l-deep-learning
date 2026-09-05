import torch
import torchvision

from torch.utils.data import Dataset

class SineData(Dataset):
    def __init__(self, T:int=1000, num_train:int=600, tau:int=4):
        self.T = T
        self.num_train=num_train
        self.tau=tau
        
        self.time = torch.arange(1, T+1, dtype=torch.float32)
        self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.time[idx], self.x[idx]

    def get_dataloader(self, train = 0):
        features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
        print(len(features))
        print(len(features[0]))
        print(type(features[0]))
        self.features = torch.stack(features,1) # Stacking elements into a new tensor along a certain dimension
        print(self.features.shape)
        self.labels = [self.x[i] for i in range(self.tau, self.T)]