import torch
import os
from basenets.mobilenet import MobileNetV2
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tensorboardX import SummaryWriter
# import argparse

def init_net():

def train(net,loaders,loss_fn,experiment_name):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--regularization', default='none', type=str,
    #                     help='none, weight_decay, dropout or batch_norm')
    # args = parser.parse_args()
    # regularization = args.regularization
    # print(regularization)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 200
    # net = Lenet5(net_params_dict, data_params_dict).to(device)
    # dataset_train = FashionDataset(os.path.join('data', 'FashionMnist'), 'train')
    # train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=1, drop_last=True)
    #
    # dataset_test = FashionDataset(os.path.join('data', 'FashionMnist'), 'test')
    # test_loader = DataLoader(dataset_test, batch_size=1000, num_workers=1)
    #
    data_loader = {"train": loaders["train"], "val": loaders["val"]}

    # loss_fn = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=weight_decay)

    # SummaryWriter encapsulates everything
    writer = SummaryWriter(os.path.join(experiment_name))

    for epoch in range(epochs):

        accuracy_dict = {}
        loss_dict = {}

        for phase, loader in data_loader.items():
            num_correct = 0
            num_samples = 0
            loss_sum = 0

            if phase == "train":
                net.train()
            else:
                net.eval()

            for i_batch, sample_batch in enumerate(loader):
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
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase == "train" and regularization == "dropout":
                    net.eval()
                    y_pred = net(images_batch)
                    net.train()

                pred_lables = torch.argmax(y_pred, 1)
                num_correct += torch.sum(torch.eq(labels_batch, pred_lables)).detach().cpu().numpy()
                num_samples += labels_batch.shape[0]
                loss_sum += loss.detach().cpu().numpy()

            avg_loss = loss_sum / len(loader)
            accuracy = num_correct / num_samples

            accuracy_dict[phase] = accuracy
            loss_dict[phase] = avg_loss

            print("loss vs epoch " + phase + ' ' + str(avg_loss) + ' ' + str(epoch))
            print("accuracy vs epoch " + phase + ' ' + str(accuracy) + ' ' + str(epoch))

        writer.add_scalars("loss vs epoch", loss_dict, epoch)
        writer.add_scalars("accuracy vs epoch", accuracy_dict, epoch)
    torch.save(net.state_dict(), regularization + ".pth")
    return regularization, device, epochs, weight_decay, net


if __name__ == '__main__':

    regularization, device, epochs, weight_decay, net = init_run()


