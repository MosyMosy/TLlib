import common.vision.datasets as datasets
import os
from torchvision.datasets.utils import download_url


def download(root: str, archive_name: str, url_link: str):
    """
    Download file from internet url link.

    Args:
        root (str) The directory to put downloaded files.
        archive_name: (str) The name of archive(zipped file) downloaded.
        url_link: (str) The url link to download data.
    """
    
    download_url(url_link, root, archive_name)
   
for i in range(1, len(datasets.__all__)):
    try:
        for file in datasets.__dict__[datasets.__all__[i]].download_list:
            download(os.path.join("~/scratch/TLlib_Dataset", datasets.__all__[i].lower()), archive_name = file[1], url_link = file[2])
    except Exception as e:
        print(e)