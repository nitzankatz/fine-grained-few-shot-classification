import torch
import os
from basenets.mobilenet import MobileNetV2
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tensorboardX import SummaryWriter
# import argparse
from utils import get_train_transforms, get_val_transforms
from torchvision import transforms, datasets
from n_way_k_shot import n_way_k_shot


def train(net, data_loader, loss_fn, experiment_name, valdir):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 50
    net = net.to(device)
    # net = Lenet5(net_params_dict, data_params_dict).to(device)
    # dataset_train = FashionDataset(os.path.join('data', 'FashionMnist'), 'train')
    # train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=1, drop_last=True)
    #
    # dataset_test = FashionDataset(os.path.join('data', 'FashionMnist'), 'test')
    # test_loader = DataLoader(dataset_test, batch_size=1000, num_workers=1)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)

    # SummaryWriter encapsulates everything
    writer = SummaryWriter(os.path.join(experiment_name))
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
            y_pred = net(images_batch)

            # Compute and print loss

            labels_batch = labels_batch.type(torch.LongTensor).to(device)
            loss = loss_fn(y_pred, labels_batch)

            if i_batch % 1000 == 0:
                print(i_batch, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.eval()
            y_pred = net(images_batch)
            net.train()

            pred_lables = torch.argmax(y_pred, 1)
            num_correct += torch.sum(torch.eq(labels_batch, pred_lables)).detach().cpu().numpy()
            num_samples += labels_batch.shape[0]
            loss_sum += loss.detach().cpu().numpy()

        avg_loss = loss_sum / len(data_loader)
        accuracy = num_correct / num_samples

        print("loss vs epoch classification train" + ' ' + str(avg_loss) + ' ' + str(epoch))
        print("accuracy vs epoch classification train " + ' ' + str(accuracy) + ' ' + str(epoch))

        writer.add_scalar("loss vs epoch", avg_loss, epoch)
        writer.add_scalar("accuracy vs epoch", accuracy, epoch)

        net.eval()
        nk = n_way_k_shot(valdir, 5, 5, net=net)
        print(nk)
        writer.add_scalar("nk vs epoch", nk, epoch)

    torch.save(net.state_dict(), ".pth")
    return device, epochs, net


if __name__ == '__main__':

    train_classes = 160
    loss_func = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    net = MobileNetV2(n_class=train_classes)

    random_state_dict = net.state_dict()
    state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth.tar'), map_location=lambda storage, loc: storage)

    state_dict['classifier.1.bias'] = random_state_dict['classifier.1.bias']
    state_dict['classifier.1.weight'] = random_state_dict['classifier.1.weight']

    net.load_state_dict(state_dict)

    # traindir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'train')
    # valdir = os.path.join('data', 'CUB_200_2011_reorganized', 'CUB_200_2011', 'images', 'val')

    traindir = os.path.join('data', 'CUB_200_2011', 'images', 'train')
    valdir = os.path.join('data', 'CUB_200_2011', 'images', 'val')

    batch_size = 32
    n_worker = 1

    input_size = 224

    train_trans_list = get_train_transforms(input_size=input_size)
    train_dataset = datasets.ImageFolder(
        traindir,train_trans_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=n_worker)  # , pin_memory=True)

    # train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1')
    train(net=net, data_loader=train_loader, loss_fn=loss_func, experiment_name='1', valdir=valdir)

    a=1


