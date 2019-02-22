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
import torch.nn.functional as F
from npairs.npairs_loss import NpairLoss
from npairs.pairs_dataloader import PairsDataSet

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
    nk_best = 0
    checkpoint = 100

    iterations = 7500

    num_samples = 0
    loss_sum = 0

    for iteration in range(iterations):

        for i_batch, sample_batch in enumerate(data_loader):
            # Forward pass: Compute predicted y by passing x to the model
            net.train()
            images_batch = sample_batch[0].to(device)
            labels_batch = sample_batch[1]
            y_pred = net.embed(images_batch)

            # Compute and print loss

            labels_batch = labels_batch.type(torch.LongTensor).to(device)
            loss = loss_fn(y_pred, labels_batch)

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

        if iteration % checkpoint == 0:
            print(i_batch, loss.item())

            avg_loss = loss_sum / checkpoint

            print("loss vs epoch classification train" + ' ' + str(avg_loss) + ' ' + str(iteration))

            writer.add_scalar("loss vs epoch", avg_loss, iteration)

            net.eval()
            nk = run_n_way_k_shot(valdir, 5, 5, net=net)
            current_nk = nk.detach().cpu().numpy()
            print(current_nk)
            writer.add_scalar("nk vs epoch", nk, iteration)

            if current_nk > nk_best:
                torch.save(net.state_dict(), os.path.join("weights", "squeezenet_classTransfer_triplet_hard_best.pth"))
                nk_best = current_nk
            torch.save(net.state_dict(), os.path.join("weights", "squeezenet_classTransfer_triplet_hard_last.pth"))

            loss_sum = 0
            num_samples = 0

    return device, epochs, net


if __name__ == '__main__':

    train_classes = 160
    loss_func = NpairLoss()
    # net = MobileNetV2(n_class=train_classes)
    net = SqueezeNet(num_classes=train_classes)

    random_state_dict = net.state_dict()
    # state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth.tar'), map_location=lambda storage, loc: storage)
    state_dict = torch.load(os.path.join('weights', 'squeezenet_classification_best_150_epochs.pth'), map_location=lambda storage, loc: storage)

    state_dict['classifier.1.bias'] = random_state_dict['classifier.1.bias']
    state_dict['classifier.1.weight'] = random_state_dict['classifier.1.weight']

    net.load_state_dict(state_dict)

    # traindir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'train')
    # valdir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'val')

    traindir = os.path.join('data', 'CUB_200_2011', 'images', 'train')
    valdir = os.path.join('data', 'CUB_200_2011', 'images', 'val')

    batch_size = 80
    n_worker = 1

    input_size = 224

    train_dataset = PairsDataSet(os.path.join('data', 'CUB_200_2011', 'images', 'train'), get_train_transforms(input_size=224))
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    # train_trans_list = get_train_transforms(input_size=input_size)
    # train_dataset = datasets.ImageFolder(
    #     traindir, train_trans_list)
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=n_worker)  # , pin_memory=True)

    # train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1')
    train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1', valdir=valdir)

    a=1