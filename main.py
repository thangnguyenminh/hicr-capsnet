import random

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

from data_loader.hicr_capsnet_data_loader import HICRCapsNetDataLoader
from trainers.hicr_capsnet_trainer import HICRCapsNetModelTrainer
from utils.config import process_config
from utils.dataset import get_label
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    """ Main Driver Program """
    # Arguments
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # Experiments
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    # Data Generators
    print("Creating data generator...")
    data_loader = HICRCapsNetDataLoader(config)

    # Some Stats & Visualizations
    x_train, y_train = data_loader.get_train_data()
    x_test, y_test = data_loader.get_test_data()
    print("Training on", len(x_train), "images")
    print("Testing on", len(x_test), "images")
    n_samples = 5
    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        idx = random.randint(0, len(x_train))
        plt.imshow(x_train[idx])
        plt.title("Label:" + get_label(y_train[idx]))
        plt.axis("off")

    plt.show()

    # Model Instance
    print("Creating the model...")

    # Trainer
    print("Creating the trainer...")
    trainer = HICRCapsNetModelTrainer(None, data_loader.get_train_data(), config)
    
    # Start training
    print("Starting to train the model...")


if __name__ == "__main__":
    main()
