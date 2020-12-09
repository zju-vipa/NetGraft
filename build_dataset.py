import os

from dataset import extract_dataset_from_cifar10, extract_dataset_from_cifar100

from torchvision.datasets.utils import download_and_extract_archive


def download_dataset():
    filename_10 = "cifar-10-python.tar.gz"
    url_10 = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_and_extract_archive(url_10, download_root='./data/', filename=filename_10, extract_root='./data/')

    filename_100 = "cifar-100-python.tar.gz"
    url_100 = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    download_and_extract_archive(url_100, download_root='./data/', filename=filename_100, extract_root='./data/')

def build_dataset():
    os.makedirs('data/', exist_ok=True)
    
    download_dataset()

    num_samples = list(range(1, 11)) + [20, 50]
    for i in num_samples:
        extract_dataset_from_cifar10(i)
        extract_dataset_from_cifar100(i)
    

if __name__ == "__main__":
    build_dataset()