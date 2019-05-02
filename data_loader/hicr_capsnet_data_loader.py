import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from base.base_data_loader import BaseDataLoader
from utils.dataset import load_numerals


class HICRCapsNetDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(HICRCapsNetDataLoader, self).__init__(config)
        (self.x, self.y) = load_numerals(config.numeral_type)
        # Dividing the dataset into 2:1 ratio, 2 for training & 1 for testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3333, random_state=42)

    def get_train_data(self):
        self.x_train = np.array(self.x_train).reshape(-1, 28, 28, 3).astype('float32') / 255.
        self.y_train = np.array(to_categorical(np.array(self.y_train).astype('float32')))
        print(self.y_train[0].shape)
        return self.x_train, self.y_train

    def get_test_data(self):
        self.x_test = np.array(self.x_test).reshape(-1, 28, 28, 3).astype('float32') / 255.
        self.y_test = np.array(to_categorical(np.array(self.y_test).astype('float32')))
        return self.x_test, self.y_test

    def get_data(self):
        self.x = np.array(self.x).reshape(-1, 28, 28, 3).astype('float32') / 255.
        self.y = np.array(to_categorical(np.array(self.y).astype('float32')))
        self.x, self.y = shuffle(self.x, self.y, random_state=42)
        return (self.x, self.y)