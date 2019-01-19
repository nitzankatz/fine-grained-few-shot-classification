import os
import numpy as np
import random

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def stanford_structuring(dir, division):

    base_dir = os.path.join(dir, 'images')
    dirs = os.listdir(base_dir)
    num_classes = len(dirs)

    random.seed(1234)
    train_inds = []

    temp = list(range(1, num_classes + 1))
    random.shuffle(temp)

    train_inds = temp[0:division['train']]
    val_inds = temp[division['train'] : division['train'] + division['validation']]
    test_inds = temp[division['train'] + division['validation'] : num_classes + 1]

    train_inds.sort()
    val_inds.sort()
    test_inds.sort()

    train_dirs = [dirs[i-1] for i in train_inds]
    val_dirs = [dirs[i-1] for i in val_inds]
    test_dirs = [dirs[i-1] for i in test_inds]


    return train_dirs, val_dirs, test_dirs


if __name__ == '__main__':
    stanford_path = os.path.join('data', 'stanford')
    stanford_division = {'train':80, 'validation':20, 'test':20}
    stanford_images_dirs = stanford_structuring(stanford_path, stanford_division)
    pass