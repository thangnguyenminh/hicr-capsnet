import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from base.base_data_loader import BaseDataLoader
from utils.dataset import load_numerals


class HICRCapsNetDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(HICRCapsNetDataLoader, self).__init__(config)
        (x, y) = load_numerals(config.numeral_type)
        # Dividing the dataset into 2:1 ratio, 2 for training & 1 for testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3333, random_state=42)

    def get_train_data(self):
        self.x_train = np.array(self.x_train).reshape(-1, 28, 28, 3).astype('float32') / 255.
        self.y_train = np.array(to_categorical(np.array(self.y_train).astype('float32')))
        return self.x_train, self.y_train

    def get_test_data(self):
        self.x_test = np.array(self.x_test).reshape(-1, 28, 28, 3).astype('float32') / 255.
        self.y_test = np.array(to_categorical(np.array(self.y_test).astype('float32')))
        return self.x_test, self.y_test
