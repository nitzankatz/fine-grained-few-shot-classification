import torch
import os
from basenets.mobilenet import MobileNetV2
from basenets.squeezenet import SqueezeNet
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tensorboardX import SummaryWriter
# import argparse
from utils import get_train_transforms, get_val_transforms
from torchvision import transforms, datasets
from n_way_k_shot import n_way_k_shot, run_n_way_k_shot
from triplet.hard_triplet_loss import HardTripletLoss
from prototipycal.proto_loss import PrototypicalLoss
from prototipycal.proto_sampler import PrototypicalBatchSampler
from prototipycal.proto_n_way_k_shot import proto_n_way_k_shot

def train(net, data_loader, loss_fn, experiment_name, valdir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 150
    net = net.to(device)
    main_tesnorboard_dir = 'logs'
    prevoius_experiments = os.listdir(main_tesnorboard_dir)
    prevoius_experiments_numeric = [int(exp_name) for exp_name in prevoius_experiments if exp_name.isdigit()]
    if len(prevoius_experiments_numeric) == 0:
        experiment_num = 1
    else:
        experiment_num = max(prevoius_experiments_numeric) + 1
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)

    # SummaryWriter encapsulates everything
    writer = SummaryWriter(os.path.join(main_tesnorboard_dir,str(experiment_num)))
    accuracy = 0

    for epoch in range(epochs):

        num_correct = 0
        num_samples = 0
        loss_sum = 0

        net.train()

        for i_batch, sample_batch in enumerate(data_loader):
            # Forward pass: Compute predicted y by passing x to the model

            images_batch = sample_batch[0].to(device)
            labels_batch = sample_batch[1]
            y_pred = net.embed(images_batch)

            # Compute and print loss

            labels_batch = labels_batch.type(torch.LongTensor).to(device)
            loss,acc = loss_fn(y_pred, labels_batch)

            if i_batch % 1000 == 0:
                print(i_batch, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # net.eval()
            # y_pred = net(images_batch)
            # net.train()

            # pred_lables = torch.argmax(y_pred, 1)

            num_samples += labels_batch.shape[0]
            loss_sum += loss.detach().cpu().numpy()

        avg_loss = loss_sum / len(data_loader)

        print("loss vs epoch classification train" + ' ' + str(avg_loss) + ' ' + str(epoch))

        writer.add_scalar("loss vs epoch", avg_loss, epoch)

        net.eval()
        # nk = run_n_way_k_shot(valdir, 2, 2, net=net)
        nk = proto_n_way_k_shot(valdir, 5, 5, net)
        print(nk.detach().cpu().numpy())
        writer.add_scalar("nk vs epoch", nk, epoch)

        torch.save(net.state_dict(), "squeezenet_proto_train.pth")
    return device, epochs, net


if __name__ == '__main__':

    train_classes = 160

    traindir = os.path.join('data', 'CUB_200_2011', 'images', 'train')
    valdir = os.path.join('data', 'CUB_200_2011', 'images', 'val')

    # traindir = r'C:\temp\tempfordeep7'
    # valdir = r'C:\temp\tempfordeep7'
    batch_size = 205
    n_worker = 1

    input_size = 224

    train_trans_list = get_train_transforms(input_size=input_size)
    train_dataset = datasets.ImageFolder(
        traindir, train_trans_list)

    classes = [sample_tuple[1] for sample_tuple in train_dataset.samples]
    sampler = PrototypicalBatchSampler(classes, 5, 40, 100)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=sampler,
        num_workers=n_worker)  # , pin_memory=True)



    loss_func = PrototypicalLoss(n_support=5)
    # net = MobileNetV2(n_class=train_classes)
    net = SqueezeNet(num_classes=train_classes)

        # random_state_dict = net.state_dict()
        # # state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth.tar'), map_location=lambda storage, loc: storage)
        # state_dict = torch.load(os.path.join('weights', 'squeezenet1_0-a815701f.pth'), map_location=lambda storage, loc: storage)
        #
        # state_dict['classifier.1.bias'] = random_state_dict['classifier.1.bias']
        # state_dict['classifier.1.weight'] = random_state_dict['classifier.1.weight']
        #
        # net.load_state_dict(state_dict)

    # traindir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'train')
    # valdir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'val')


    # train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1')
    train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1', valdir=valdir)

    a=1