import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import get_val_transforms

def main(root_dir, N, k):
    classes_dirs = get_n_classes_dirs(N, root_dir)
    comparable_lists, test_lists = split_classes(classes_dirs, k)
    acc = get_accuracy(comparable_lists, test_lists)
    a = 3


def get_accuracy(comparable_lists, test_lists):
    total_correct, total_samples = 0, 0
    comparable_images = get_images_for_lists(comparable_lists)
    for class_num, class_test_list in enumerate(test_lists):
        num_correct, num_samples = test_class(class_test_list, class_num, comparable_images)
        total_correct += num_correct
        total_samples += num_samples
    return total_correct / total_samples


def get_images_for_lists(lists):
    for list in lists[1:]:
        assert len(lists[0]) == len(list)
    classes_batches = [list_to_batch(list) for list in lists]
    images = torch.cat(classes_batches)
    return images


def list_to_batch(list):
    dataset = ListDataset(list,get_val_transforms(224))
    loader = DataLoader(dataset,batch_size=len(dataset))
    return next(iter(loader))



def test_class(class_test_lists, class_num, comparable_images):
    return 1, len(class_test_lists)


def get_n_classes_dirs(N, root_dir):
    classes_dirs = my_list_dir(root_dir)
    classes_dirs, _ = choose_n_from_list(classes_dirs, N)
    return classes_dirs


def split_classes(classes_dirs, k):
    comparable_lists = []
    test_lists = []
    for class_dir in classes_dirs:
        class_comparables, class_test = split_class(class_dir, k)
        comparable_lists.append(class_comparables)
        test_lists.append(class_test)
    return comparable_lists, test_lists


def split_class(class_dir, k):
    class_images = my_list_dir(class_dir)
    class_comparables, class_test = choose_n_from_list(class_images, k)
    return class_comparables, class_test


def my_list_dir(root_dir):
    subs = os.listdir(root_dir)
    return [os.path.join(root_dir, sub) for sub in subs]


def choose_n_from_list(list, n):
    assert n < len(list)
    random.seed(1234)
    random.shuffle(list)
    n_list = list[:n]
    rest_list = list[n:]
    return n_list, rest_list


class ListDataset(Dataset):
    def __init__(self, path_list, transform):
        super(ListDataset, self).__init__()
        self.list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        with open(self.list[idx], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return self.transform(img)


if __name__ == '__main__':
    root_dir = r"C:\temp\tempfordeep"
    main(root_dir, 3, 1)
