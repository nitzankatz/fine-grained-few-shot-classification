from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
from utils import get_val_transforms
from basenets.squeezenet import SqueezeNet
from npairs.npairs_loss import NpairLoss

def get_files_list(base_dir, class_dir):
    class_path = os.path.join(base_dir, class_dir)
    files_names = os.listdir(class_path)
    return [os.path.join(class_path, file_name) for file_name in files_names]

class PairsDataSet(Dataset):
    def __init__(self, base_dir, transform):
        super(Dataset, self).__init__()
        self.transform = transform
        self.classes = os.listdir(base_dir)
        self.samples = [get_files_list(base_dir, class_dir) for class_dir in self.classes]

    def get_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            return self.transform(img)

    def __getitem__(self, idx):
        if idx >= len(self.classes):
            raise Exception('idx should be 0<= idx < len(classes)')
        curr_class_samples = self.samples[idx]
        inds_in_class = random.sample(range(len(curr_class_samples)), 2)
        anchor = self.get_image(curr_class_samples[inds_in_class[0]])
        positive = self.get_image(curr_class_samples[inds_in_class[1]])
        label = idx
        return anchor, positive, label

    def __len__(self):
        return len(self.classes)


if __name__ == '__main__':
    net = SqueezeNet()
    loss_func = NpairLoss()
    p = PairsDataSet(os.path.join('data', 'CUB_200_2011', 'images', 'val'), get_val_transforms(input_size=224))
    loader = DataLoader(p, batch_size=210, shuffle=True)
    for x in loader:
        embed0 = net.embed(x[0])
        embed1 = net.embed(x[1])
        loss = loss_func(embed0,embed1,x[2])
        print(loss)

    pass
