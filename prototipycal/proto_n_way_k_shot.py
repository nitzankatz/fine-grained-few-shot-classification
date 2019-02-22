import torch
from utils import get_val_transforms
from torchvision import datasets
from prototipycal.proto_sampler import PrototypicalBatchSampler
from prototipycal.proto_loss import PrototypicalLoss
from basenets.squeezenet import SqueezeNet
import os
import sys


def proto_n_way_k_shot(val_dir, N, k, net, input_size=224, num_support=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    val_trans_list = get_val_transforms(input_size=input_size)
    val_dataset = datasets.ImageFolder(
        val_dir, val_trans_list)
    classes = [sample_tuple[1] for sample_tuple in val_dataset.samples]
    sampler = PrototypicalBatchSampler(classes, N, k + num_support, 600)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=1, batch_sampler=sampler)  # , pin_memory=True)
    loss_func = PrototypicalLoss(n_support=5)
    sum_acc = 0
    num_acc = 0
    for x, y in iter(val_loader):
        emb = net.embed(x.to(device))
        loss, acc = loss_func(emb, y.to(device))
        sum_acc += acc
        num_acc += 1
    return (sum_acc / num_acc)


if __name__ == '__main__':
    root_dir = os.path.join('data', 'CUB_200_2011', 'images', 'test')
    # weight_path = 'random'
    # weight_path = os.path.join('weights', 'squeezenet_class.pth')
    weight_path = sys.argv[1]

    net = SqueezeNet()
    state_dict = torch.load(weight_path,
                            map_location=lambda storage, loc: storage)

    net.load_state_dict(state_dict)
    net.eval()
    acc = proto_n_way_k_shot(root_dir, 5, 5, net)
    print(acc)
