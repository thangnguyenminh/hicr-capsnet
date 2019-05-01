import argparse
import csv
from glob import glob
import os
import re
from shutil import copyfile

import numpy as np
from PIL import Image

DIMS = (28, 28)

def load_numerals(numeral_type):
    """ Loads training & testing images for given type of numeral 
    Args:
		numeral_type (string): type of numeral to load
	Returns:
		(x, y): dataset
    """
    # Relative path things
    rel_dirname = os.path.dirname(__file__)

    # Entries
    x, y = [], []

    # List all directories & files inside the same
    for dirname in os.listdir(os.path.join(rel_dirname, '../data/training/'+numeral_type)):
        for filename in glob(os.path.join(rel_dirname, '../data/training/'+numeral_type+'/'+dirname+'/*.bmp')):
            img = Image.open(os.path.join(rel_dirname, filename))
            img = img.resize(DIMS)
            x.append(np.array(img).flatten())
            y.append(str(dirname))

    return (x, y)


def get_label(data):
  for i in range(0, len(data)):
    if data[i] == 1:
      return str(i)