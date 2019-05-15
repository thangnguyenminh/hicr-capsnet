# hicr-capsnet
Tensorflow 2.0 implementation of "Handwritten Indic Character Recognition using Capsule Networks" [ASPCON 2018]

## Initial Setup
1. Setup the environment & install all the dependencies:  
``pip install -r requirements.txt``
2. Download the datasets [Bangla Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.1.1.rar), [Telugu Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.4.1.rar), [Devanagari Digits](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/cmaterdb/CMATERdb%203.2.1.rar) and rename as **bangla, devanagari, telugu** in ``data`` directory
3. ``python data/process_dataset.py``
4. If you want to create csv file from images folder then: 
``python data/process_dataset.py --csv``

## Running
- For starting training & validation:  
``python main.py``

## Acknowledgements
A major thanks to authors of the paper.   

## Reference
    @misc{1901.00166,
    Author = {Bodhisatwa Mandal and Suvam Dubey and Swarnendu Ghosh and Ritesh Sarkhel and Nibaran Das},
    Title = {Handwritten Indic Character Recognition using Capsule Networks},
    Year = {2019},
    Eprint = {arXiv:1901.00166},
    }