import numpy as np
import torch
from utils import get_val_transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from basenets.mobilenet import MobileNetV2
import os
from prototipycal.proto_loss import PrototypicalLoss
from basenets.squeezenet import SqueezeNet


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


if __name__ == '__main__':
    classes = range(50)
    input_size = 224
    val_dir = r'C:\dev\studies\deepLearning\fine-grained-few-shot-calssification\data\CUB_200_2011\images\val'
    val_trans_list = get_val_transforms(input_size=input_size)
    val_dataset = datasets.ImageFolder(
        val_dir, get_val_transforms(input_size))
    classes = [sample_tuple[1] for sample_tuple in val_dataset.samples]
    sampler = PrototypicalBatchSampler(classes, 5, 10, 100)

    # net = MobileNetV2()
    # state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth'), map_location=lambda storage, loc: storage)

    net = SqueezeNet(num_classes=160)
    # weights_path = os.path.join('weights', 'squeezenet1_0-a815701f.pth')
    weights_path = r"C:\temp\weights\classification.pth"
    state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)

    net.load_state_dict(state_dict)
    net.eval()

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
        print(acc)
    print("final: " + str(sum_acc / num_acc))

    #
    # for i , b in enumerate(val_loader):
    #     pass
