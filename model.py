import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))  # hidden layer 1
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class TwoHiddenLayerNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, output_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
        self.activation1 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))  # hidden layer 1
        x = self.activation(self.fc2(x))  # hidden layer 2
        x = self.fc3(x)                   # output layer (no activation here)
        return x
