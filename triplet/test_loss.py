import torch
from basenets.mobilenet import MobileNetV2
from triplet.hard_triplet_loss import HardTripletLoss
from torchvision import transforms, datasets
from torch.utils.data import dataloader
from triplet.hard_triplet_loss import HardTripletLoss
import os


def main():
    loss_func = HardTripletLoss(hardest=True)
    batch = torch.zeros([5, 3, 200, 200])
    labels = torch.tensor([1, 3, 3, 4, 4])
    net = MobileNetV2()
    state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth'), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    embeddings = net.embed(batch)
    loss = loss_func(embeddings, labels)
    a = 3
    traindir = r'C:\temp\tempfordeep'
    valdir = r'C:\temp\tempfordeep'
    batch_size = 4
    n_worker = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    input_size = 224
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=n_worker)#, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=n_worker)#, pin_memory=True)
    train_in = next(iter(train_loader))
    val_in = next(iter(val_loader))
    train_im, train_class = train_in[0], train_in[1]
    val_im, val_class = val_in[0], val_in[1]

    embed_train = net.embed(train_im)
    loss = loss_func(embed_train,train_class)
    loss.backward()
    net.eval()
    embed_val = net.embed(val_im)
    loss = loss_func(embed_val,val_class)
    # predicted_classes = torch.argmax(out_val)
    # topk, predicted_classes = torch.topk(out_val.squeeze(), 5)
    a = 3
if __name__ == '__main__':
    main()