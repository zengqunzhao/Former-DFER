import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, 7)

    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
