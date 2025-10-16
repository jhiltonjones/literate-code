import torch 
import torch.nn as nn
import joblib
import torch.optim as optim
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16,16),
            nn.Tanh(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.net(x)
    

