import torch
import os
import random


def main(root_dir, N, k):
    classes_dirs = get_n_classes_dirs(N, root_dir)
    comparable_lists, test_lists = split_classes(classes_dirs, k)
    total_correct, total_samples = 0, 0
    for class_num, class_test_images in enumerate(test_lists):
        num_correct, num_samples = test_class(class_test_images, class_num, comparable_lists)
        total_correct += num_correct
        total_samples += num_samples
    acc = total_correct / total_samples
    a=3


def test_class(class_test_images, class_num, comparable_lists):
    return 1, len(class_test_images)


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


if __name__ == '__main__':
    root_dir = r"C:\temp\tempfordeep"
    main(root_dir, 3, 1)
