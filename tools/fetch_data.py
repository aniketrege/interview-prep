"""Download every dataset the drills use into prep.DATA (idempotent)."""
import urllib.request

import sklearn.datasets as skd
import torchvision

from prep import DATA

DATA.mkdir(parents=True, exist_ok=True)
for cls in (torchvision.datasets.MNIST, torchvision.datasets.FashionMNIST, torchvision.datasets.CIFAR10):
    for train in (True, False):
        print(cls.__name__, "train" if train else "test", len(cls(DATA, train=train, download=True)))
print("california housing", skd.fetch_california_housing(data_home=DATA / "sklearn").data.shape)
(DATA / "ag_news").mkdir(exist_ok=True)
for split in ("train", "test"):
    dst = DATA / "ag_news" / f"{split}.parquet"
    if not dst.exists():  # https://huggingface.co/datasets/fancyzhx/ag_news
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/{split}-00000-of-00001.parquet", dst)
    print("ag_news", split, dst.stat().st_size, "bytes")
