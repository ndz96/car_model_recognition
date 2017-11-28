from scipy.io import loadmat
from scipy import misc
from PIL import Image
import numpy as np
import os


class BMW:

    def __init__(self, batch_size):
        self.train_pics = []
        self.train_labels = []
        self.val_pics = []
        self.val_labels = []
        self.batch_size = batch_size
        self.index_train = 0
        self.index_val = 0
        self.epochs_done = 0
        self.num_train_examples = 0
        self.num_val_examples = 0

    def load(self, image_size):
        bmw = loadmat('bmw10_annos.mat')
        str = os.getcwd() + "\\bmw10_ims"
        train_data = []
        for example in bmw['annos'][0]:
            label = example[1][0][0] - 1
            im_path = example[0][0]
            pic = misc.imread(str + "\\" + im_path)
            # print(pic.shape)
            # img = Image.fromarray(pic, 'RGB')
            # img.save('my1.png')
            # img.show()
            # input("")
            pic = misc.imresize(pic, (image_size, image_size, 3))
            # print(pic.shape)
            # img = Image.fromarray(pic, 'RGB')
            # img.save('my.png')
            # img.show()
            # input("")
            train_data.append((pic, label))
        np.random.shuffle(train_data)

        validation_size = int(len(train_data) * 0.20)
        val_data = train_data[:validation_size].copy()
        np.random.shuffle(val_data)
        train_data = train_data[validation_size:]

        for i in range(len(train_data)):
            self.train_pics.append(train_data[i][0])
            self.train_labels.append(train_data[i][1])
        for i in range(len(val_data)):
            self.val_pics.append(val_data[i][0])
            self.val_labels.append(val_data[i][1])
        # print(len(self.train_pics), len(self.val_pics))
        self.num_train_examples = len(self.train_pics)
        self.num_val_examples = len(self.val_pics)

    def get_train_batch(self):
        start = self.index_train
        self.index_train += self.batch_size

        if self.index_train > self.num_train_examples:
            self.epochs_done += 1
            start = 0
            self.index_train = self.batch_size
        end = self.index_train

        return self.train_pics[start:end], self.train_labels[start:end]

    def get_val_batch(self):
        start = self.index_val
        self.index_val += self.batch_size

        if self.index_val > self.num_val_examples:
            start = 0
            self.index_val = self.batch_size
        end = self.index_val

        return self.val_pics[start:end], self.val_labels[start:end]
    
    def get_train_size(self):
        return self.num_train_examples

    def get_val_size(self):
        return self.num_val_examples

# bmw = BMW(16)
# bmw.load(256)





