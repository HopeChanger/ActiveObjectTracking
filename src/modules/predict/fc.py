import torch.nn as nn
import torch.nn.functional as F


class FCPredictNetwork(nn.Module):
    def __init__(self, input_shape=440, output_shape=44):
        super(FCPredictNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return x + inputs[:, -44:]
