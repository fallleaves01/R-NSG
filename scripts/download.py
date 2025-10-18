import os
import tarfile
import urllib.request

dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
os.makedirs(dataset_dir, exist_ok=True)

files = {
    "siftsmall.tar.gz": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    # "sift.tar.gz": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
}

for filename, url in files.items():
    dest_path = os.path.join(dataset_dir, filename)
    print(f"Downloading {filename} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")

for filename in files:
    tar_path = os.path.join(dataset_dir, filename)
    print(f"Extracting {filename} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dataset_dir)