import common.vision.datasets as datasets
import os.path as osp

for dataset in datasets.__all__:
    try:
        _ = datasets.__dict__[dataset](osp.join("~/scratch/TLlib_Dataset", dataset.lower()), download=True)
    except:
        print("couldn't download the {}".format(dataset))
        
        