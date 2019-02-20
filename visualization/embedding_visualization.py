import torch
from basenets.squeezenet import SqueezeNet
from torchvision import datasets
from sklearn.manifold import TSNE
from utils import get_val_transforms
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_tsne_embeddings(val_dir, weights_path):
    input_size = 224
    train_classes = 160
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SqueezeNet(num_classes=train_classes)
    net.to(device)
    if weights_path != 'random':
        random_state_dict = net.state_dict()
        state_dict = torch.load(weights_path,
                                map_location=lambda storage, loc: storage)
        state_dict['classifier.1.bias'] = random_state_dict['classifier.1.bias']
        state_dict['classifier.1.weight'] = random_state_dict['classifier.1.weight']
        net.load_state_dict(state_dict)
    else:
        weights_path = os.path.join('weights', 'random')
    val_trans_list = get_val_transforms(input_size=input_size)
    val_dataset = datasets.ImageFolder(
        val_dir, val_trans_list)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=1, batch_size=10, shuffle=True)
    all_embeds = None
    all_labels = None
    for i, (batch, labels) in enumerate(val_loader):
        # if i == 10:
        #     break
        current_embeds = net.embed(batch.to(device)).detach().cpu().numpy()
        if i == 0:
            all_embeds = current_embeds
            all_labels = labels
        else:
            # all_embeds = torch.cat((all_embeds, current_embeds), 0)
            # all_labels = torch.cat((all_labels, labels), 0)
            all_embeds = np.concatenate((all_embeds, current_embeds), 0)
            all_labels = np.concatenate((all_labels, labels), 0)

    # numpy_embeds = all_embeds.detach().cpu().numpy()
    # numpy_labels = all_labels.detach().cpu().numpy()
    numpy_embeds = all_embeds
    numpy_labels = all_labels
    # labels_msb, labels_lsb =  np.divmod(numpy_labels,len(markers))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(numpy_embeds)
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, title='tsne')
    # Create the scatter

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    markers = [".", ",", "o", "v", "^", "<", ">"]
    assert len(markers) == len(colors)
    div_value = len(markers)
    for sample_num in range(tsne_results.shape[0]):
        ax.scatter(
            x=tsne_results[sample_num, 0],
            y=tsne_results[sample_num, 1],
            c=colors[numpy_labels[sample_num] % div_value],
            marker=markers[numpy_labels[sample_num] // div_value],
            cmap=plt.cm.get_cmap('Paired'),
            alpha=0.9)
    # fig.show()
    plt.savefig(weights_path.split('.')[0] + '.png')
    a = 3


if __name__ == '__main__':
    valdir = os.path.join('data', 'CUB_200_2011', 'images', 'val')
    # weight_path = 'random'
    weight_path = os.path.join('weights', 'squeezenet_class.pth')
    plot_tsne_embeddings(valdir, weight_path)
