# hicr-capsnet
Tensorflow implementation of "Handwritten Indic Character Recognition using Capsule Networks" [ASPCON 2018]

## Initial Setup
1. Setup the environment & install all the dependencies:  
``pip install -r requirements.txt``
2. Download the datasets [Bangla Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.1.1.rar), [Telugu Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.4.1.rar), [Devanagari Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.2.1.rar) and rename as **bangla, devanagari, telugu** in ``data`` directory
3. ``python data/process_dataset.py``
4. If you want to create csv file from images folder then: 
``python data/process_dataset.py --csv``

## Running
1. For starting training & validation
``python main.py -c configs/hicr_capsnet_config.json``
2. If you want to see some sample images before training
``python main.py -c configs/hicr_capsnet_config.json -s``

## Code Description
1. ``base``: Contains the base model and base trainer. The model and trainer are inherited from here.
2. ``configs/``:  Contains the configuration files stored in json format. All hyper-parameters are stored here.
3. ``data/``: Contains scripts to process the data files and store them into numpy format
4. ``data_loader/``: Contains the custom class which is used to get data from the pipeline.
5. ``models/``: Contains model definitions, each of which is a class. There is only 1 such class as of now: CapsNet.
6. ``trainers/``: Contains the trainer, which schedules the training, evaluation, tensorboard logging among different things.
7. ``utils/``: Contains utility function like the process_config which is used to parse the configuration file.

## Acknowledgements
A major thanks to authors of the paper.   

## Reference
    @misc{1901.00166,
    Author = {Bodhisatwa Mandal and Suvam Dubey and Swarnendu Ghosh and Ritesh Sarkhel and Nibaran Das},
    Title = {Handwritten Indic Character Recognition using Capsule Networks},
    Year = {2019},
    Eprint = {arXiv:1901.00166},
    }