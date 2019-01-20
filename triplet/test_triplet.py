from basenets.mobilenet import MobileNetV2
import torch
import os
from triplet import losses
from triplet.triplet_net import TripletNet
from triplet.triplet_utils import HardestNegativeTripletSelector
# loss_func = losses.TripletLoss(1.0)
loss_func = losses.OnlineTripletLoss(1.0,HardestNegativeTripletSelector(1.0))

basenet = MobileNetV2()
net = TripletNet(basenet)
state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth'), map_location=lambda storage, loc: storage)
net.load_weights(state_dict)
a = torch.rand([5, 3, 200, 200])
p = torch.rand([5, 3, 200, 200])
n = torch.rand([5, 3, 200, 200])
o1, o2, o3 = net(a, p, n)
loss = loss_func(o1, o2, o3)
b = 3
