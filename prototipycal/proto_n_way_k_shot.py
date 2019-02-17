import torch
from utils import get_val_transforms
from torchvision import datasets
from prototipycal.proto_sampler import PrototypicalBatchSampler
from prototipycal.proto_loss import PrototypicalLoss


def proto_n_way_k_shot(val_dir, N, k,net, input_size=224, num_support=16):
    val_trans_list = get_val_transforms(input_size=input_size)
    val_dataset = datasets.ImageFolder(
        val_dir, val_trans_list)
    classes = [sample_tuple[1] for sample_tuple in val_dataset.samples]
    sampler = PrototypicalBatchSampler(classes, N, k + num_support, 100)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=1, batch_sampler=sampler)  # , pin_memory=True)
    loss_func = PrototypicalLoss(n_support=1)
    sum_acc = 0
    num_acc = 0
    for x, y in iter(val_loader):
        emb = net.embed(x)
        loss, acc = loss_func(emb, y)
        sum_acc += acc
        num_acc += 1
    return (sum_acc / num_acc)
