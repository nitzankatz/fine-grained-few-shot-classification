from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_val_transforms(input_size):
    return transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_train_transforms(input_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


def image_loader(image_name,transform):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU
