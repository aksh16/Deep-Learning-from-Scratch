import numpy as np
import matplotlib.pyplot as plt
import logging
from random import shuffle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("General Logger")


class Dataset(object):
    def __init__(self, imgs: np.ndarray, labels: np.ndarray, size: int, dims: tuple[int, int]) -> None:
        self.imgs = imgs
        self.labels = labels
        self.size = size
        self.dims = dims

    def shuffle(self):
        img_label = list(zip(self.labels, self.imgs))
        shuffle(img_label)
        self.imgs = [img[1] for img in img_label]
        self.labels = [img[0] for img in img_label]
        log.info("Shuffled training data")
        return

    def visualize(self, num: int):
        for i in range(num):
            img = self.imgs[i].reshape(28, 28)
            label = self.labels[i]
            plt.imshow(img, interpolation='nearest')
            log.info("image label:{}".format(label))
            plt.show()
        return

    def split(self, percent: float):
        print('split', type(self.imgs))
        x_imgs = self.imgs[:int(percent * self.size)]
        x_labels = self.labels[:int(percent * self.size)]
        dataset = Dataset(imgs=x_imgs, labels=x_labels, size=int(percent * self.size), dims=self.dims)
        self.imgs = self.imgs[int(percent * self.size):]
        self.labels = self.labels[int(percent * self.size):]
        self.size = self.size - int(percent * self.size)
        log.info("Shuffled and split training data")
        return dataset
