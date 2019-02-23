import torch
from utils import get_val_transforms
from torchvision import datasets
from prototipycal.proto_sampler import PrototypicalBatchSampler
from prototipycal.proto_loss import PrototypicalLoss
from basenets.squeezenet import SqueezeNet
import os
import sys


def proto_n_way_k_shot(val_dir, N, k, net, input_size=224, num_query=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    val_trans_list = get_val_transforms(input_size=input_size)
    val_dataset = datasets.ImageFolder(
        val_dir, val_trans_list)
    classes = [sample_tuple[1] for sample_tuple in val_dataset.samples]
    sampler = PrototypicalBatchSampler(classes, N, k + num_query, 600)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=1, batch_sampler=sampler)  # , pin_memory=True)
    loss_func = PrototypicalLoss(n_support=k)
    # sum_acc = 0
    # num_acc = 0
    acc_list = []
    counter = 0
    for x, y in iter(val_loader):
        emb = net.embed(x.to(device))
        loss, acc = loss_func(emb, y.to(device))
        acc_list.append(acc)
        counter += 1
        if counter % 100 == 0:
            print(counter)
    accs = torch.stack(acc_list)
    return torch.mean(accs), torch.std(accs)


if __name__ == '__main__':
    root_dir = os.path.join('data', 'CUB_200_2011', 'images', 'test')
    # weight_path = 'random'
    # weight_path = os.path.join('weights', 'squeezenet_class.pth')
    weight_path = sys.argv[1]

    net = SqueezeNet()
    random_state_dict = net.state_dict()
    state_dict = torch.load(weight_path,
                            map_location=lambda storage, loc: storage)
    state_dict['classifier.1.bias'] = random_state_dict['classifier.1.bias']
    state_dict['classifier.1.weight'] = random_state_dict['classifier.1.weight']
    net.load_state_dict(state_dict)
    net.eval()
    mean, std = proto_n_way_k_shot(root_dir, 5, 5, net)
    print('mean: ' + str(mean))
    print('std: ' + str(std))
