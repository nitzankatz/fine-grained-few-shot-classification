# fine-grained-few-shot-classification
a pytorch implementation to fine-grained few shot classification using triplet loss

Fine-grained classification is a sub genre of the classificatiom problem, focused on facing with datasets with subtle diffrences between classes. Few-shot classification is problem where the classifier can learn from some seen classes, but later is tested on classes with only few samples to learn from. In our research we studied differenet approches to attack the combined problem. Our work is focused on using variants of triplet loss, and different settings of transfer learning between methods. For fast runs, we used Squeezenet as our backbone, with performance compromise. We experimented our methods on Caltech's CUB 200 dataset and measured ourself with N-way K-shot accuracy. Good results can be achieved with Imagenet only pretrained weights, on a relatively small netwrok such as Squeezenet. Our main contribution is using transfer learning between methods to improve the N-way K-shot accuracy as well as using popular triplet loss from face recognition domain to fine-grained few-shot classification.

## inspiration
squeezenet code taken from:
https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

mobilenet taken from
https://github.com/marvis/pytorch-mobilenet

triplet loss hadnling code taken from:
https://github.com/lyakaap/NetVLAD-pytorch
(https://github.com/adambielski/siamese-triplet)

a closer look on few shot classification
https://github.com/wyharveychen/CloserLookFewShot
(https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch)

n-pairs loss
https://github.com/ChaofWang/Npair_loss_pytorch

