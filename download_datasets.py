import common.vision.datasets as datasets
import os
from torchvision.datasets.utils import download_url


for dataset_name in datasets.__all__:
    dataset =  datasets.__dict__[dataset_name](root="", download=False)
    list(map(lambda args: download(os.path.join("~/scratch/TLlib_Dataset", dataset_name.lower()), *args),dataset.download_list))
            
def download(root: str, name, archive_name: str, url_link: str):
    """
    Download file from internet url link.

    Args:
        root (str) The directory to put downloaded files.
        archive_name: (str) The name of archive(zipped file) downloaded.
        url_link: (str) The url link to download data.
    """
    
    try:
        download_url(url_link, download_root=root, filename=archive_name)
    except Exception:
        print("Fail to download {} from url link {}".format(archive_name, url_link))
        print('Please check you internet connection.'
                "Simply trying again may be fine.")
        exit(0)
            
