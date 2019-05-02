import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

from data_loader.hicr_capsnet_data_loader import HICRCapsNetDataLoader
from models.capsnet_model import CapsNetModel
from trainers.hicr_capsnet_trainer import HICRCapsNetModelTrainer
from utils.config import process_config
from utils.dataset import get_label
from utils.dirs import create_dirs
from utils.utils import get_args

tf.logging.set_verbosity(tf.logging.ERROR)

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
    x, y = data_loader.get_data()
    print("Training & Validation on", len(x), "images")
    if args.show_img:
        n_samples = 5
        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(1, n_samples, index + 1)
            idx = random.randint(0, len(x))
            plt.imshow(x[idx])
            plt.title("Label:" + get_label(y[idx]))
            plt.axis("off")

        plt.show()

    # Model Instance
    print("Creating the model...")
    model = CapsNetModel(config)

    # Trainer
    print("Creating the trainer...")
    trainer = HICRCapsNetModelTrainer(model.model, ([x, y], [y, x]), config)
    
    # Start training
    print("Starting to train the model...")
    trainer.train()

if __name__ == "__main__":
    main()
