import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import get_val_transforms
from basenets.mobilenet import MobileNetV2
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_check(root_dir, N, k, seed):
    classes_dirs = get_n_classes_dirs(N, root_dir, seed)
    comparable_lists, test_lists = split_classes(classes_dirs, k, seed)
    print(comparable_lists)


def n_way_k_shot(root_dir, N, k, net):
    classes_dirs = get_n_classes_dirs(N, root_dir)
    # comparable_lists, test_lists = split_classes(classes_dirs, k)
    comparable_lists, test_lists = mock_split_classes3()
    return get_accuracy(comparable_lists, test_lists, net)


def mock_split_classes1():
    comparable_lists = [
        [r"C:\temp\tempfordeep4\0\icecream1.jpg"],
        [r"C:\temp\tempfordeep4\1\bus.jpg"]
    ]
    test_lists = [
        [
            r"C:\temp\tempfordeep4\0\icecream1 - Copy.jpg",
            r"C:\temp\tempfordeep4\0\bus3.jpg",
            r"C:\temp\tempfordeep4\0\bus2.jpg",
            r"C:\temp\tempfordeep4\0\bus1.jpg",
            r"C:\temp\tempfordeep4\0\bus3.jpg"
        ],
        [
            r"C:\temp\tempfordeep4\1\bus2.jpg",
            r"C:\temp\tempfordeep4\1\ice1.jpg",
            r"C:\temp\tempfordeep4\1\ice2.jpg",
            r"C:\temp\tempfordeep4\1\ice3.jpg"
        ]
    ]
    return comparable_lists, test_lists


def mock_split_classes2():
    comparable_lists = [
        [r"C:\temp\tempfordeep5\0\Painted_Bunting_0006_15249.jpg"],
        [r"C:\temp\tempfordeep5\1\Ivory_Gull_0093_49052 - Copy.jpg"]
    ]
    test_lists = [
        [
            r"C:\temp\tempfordeep5\0\Painted_Bunting_0006_15249 - Copy.jpg",
            r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996.jpg",
            r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996 - Copy.jpg",
            r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996 - Copy (2).jpg"
        ],
        [
            r"C:\temp\tempfordeep5\1\Ivory_Gull_0093_49052.jpg",
            r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530 - Copy (2).jpg",
            r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530 - Copy.jpg",
            r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530.jpg"
        ]
    ]
    return comparable_lists, test_lists


def mock_split_classes3():
    comparable_lists = [
        [r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996 - Copy (2).jpg"],
        [r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530.jpg"]
    ]
    test_lists = [
        [r"C:\temp\tempfordeep5\0\Painted_Bunting_0006_15249.jpg",
         r"C:\temp\tempfordeep5\0\Painted_Bunting_0006_15249 - Copy.jpg",
         r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996.jpg",
         r"C:\temp\tempfordeep5\0\Ivory_Gull_0092_49996 - Copy.jpg"

         ],
        [r"C:\temp\tempfordeep5\1\Ivory_Gull_0093_49052 - Copy.jpg",
         r"C:\temp\tempfordeep5\1\Ivory_Gull_0093_49052.jpg",
         r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530 - Copy (2).jpg",
         r"C:\temp\tempfordeep5\1\Painted_Bunting_0029_16530 - Copy.jpg"
         ]
    ]
    return comparable_lists, test_lists


def embed_images(images, net):
    raw_embeddings = net.embed(images.to(device))
    return F.normalize(raw_embeddings)


def get_accuracy(comparable_lists, test_lists, net):
    total_correct, total_samples = 0, 0
    comparable_embeddings = get_embeddings_for_lists(comparable_lists, net)
    for class_num, class_test_list in enumerate(test_lists):
        num_correct, num_samples = test_class(class_test_list, class_num, comparable_embeddings, net)
        total_correct += num_correct
        total_samples += num_samples
    return total_correct.float().div(total_samples)


def get_embeddings_for_lists(lists, net):
    for list in lists[1:]:
        assert len(lists[0]) == len(list)
    classes_batches = [torch.unsqueeze(embed_images(list_to_batch(list), net), 0) for list in lists]
    images = torch.cat(classes_batches)
    return images


def list_to_batch(list):
    batch_size = len(list)
    loader = get_list_loader(list, batch_size)
    return next(iter(loader))


def get_list_loader(list, batch_size):
    dataset = ListDataset(list, get_val_transforms(224))
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def test_class(class_test_list, class_num, comparable_embeddings, net):
    test_loader = get_list_loader(class_test_list, 64)
    # test_loader = get_list_loader(class_test_list, 1)
    comparable_embeddings = comparable_embeddings.unsqueeze(2)
    num_correct = 0
    for batch_num, images in enumerate(test_loader):
        embeddings = embed_images(images, net)
        embeddings = embeddings.unsqueeze(0).unsqueeze(1)
        diff = embeddings - comparable_embeddings
        num_correct += get_num_correct(class_num, diff)

    return num_correct, len(class_test_list)


def get_num_correct(class_num, diff):
    dists = torch.norm(diff, dim=3)
    sum_dists = dists.sum(1)
    tmp = sum_dists.detach().cpu().numpy()
    max_class = torch.argmin(sum_dists, dim=0)
    correct = max_class == class_num
    return correct.sum()


def get_n_classes_dirs(N, root_dir, seed=1234):
    classes_dirs = my_list_dir(root_dir)
    classes_dirs, _ = choose_n_from_list(classes_dirs, N, seed)
    return classes_dirs


def split_classes(classes_dirs, k, seed=1234):
    comparable_lists = []
    test_lists = []
    for class_dir in classes_dirs:
        class_comparables, class_test = split_class(class_dir, k, seed)
        comparable_lists.append(class_comparables)
        test_lists.append(class_test)
    # test_lists.reverse()
    return comparable_lists, test_lists


def split_class(class_dir, k, seed=1234):
    class_images = my_list_dir(class_dir)
    class_comparables, class_test = choose_n_from_list(class_images, k, seed)
    return class_comparables, class_test


def my_list_dir(root_dir):
    subs = os.listdir(root_dir)
    return [os.path.join(root_dir, sub) for sub in subs]


def choose_n_from_list(list, n, seed):
    assert n <= len(list)
    random.seed(seed)
    # random.seed(1249)
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
    # root_dir = r"C:\temp\tempfordeep"
    # root_dir = r'C:\dev\studies\deepLearning\fine-grained-few-shot-calssification\data\CUB_200_2011\images\val'
    root_dir = r'C:\temp\tempfordeep4'
    # for seed in range(200):
    #     print(seed)
    #     split_check(root_dir, 2, 1,seed)
    net = MobileNetV2()
    state_dict = torch.load(os.path.join('weights', 'mobilenet_v2.pth'), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    net.eval()
    acc = n_way_k_shot(root_dir, 2, 1, net)
    print("acc is: " + str(acc))
