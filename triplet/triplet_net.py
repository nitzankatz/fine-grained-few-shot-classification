from torch import nn
import torch


class TripletNet(nn.Module):
    def __init__(self, base_net):
        super(TripletNet, self).__init__()
        self.base_net = base_net

    def load_weights(self, base_net_state_dict):
        self.base_net.load_state_dict(base_net_state_dict)

    def save_weights(self, path):
        torch.save(self.base_net.state_dict(), path)

    def forward(self, x1, x2, x3):
        output1 = self.base_net(x1)
        output2 = self.base_net(x2)
        output3 = self.base_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.base_net(x)
