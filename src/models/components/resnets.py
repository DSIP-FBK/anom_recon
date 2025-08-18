
from torch import nn

class ResnetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResnetBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

        self.projection = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + self.projection(residual)
        return x

class ConvResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()

        self.projection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + self.projection(residual)
        return x