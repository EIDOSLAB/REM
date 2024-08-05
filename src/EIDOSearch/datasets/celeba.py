import os
import pandas as pd
import gdown
import PIL
import tarfile
import torch

from EIDOSearch.utilities import ensure_dir


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split='train', target='Male', transform=None, seed=42):
        path = root
        ensure_dir(path)

        if not os.path.isdir(os.path.join(path, 'CelebA')):
            self.download_dataset(path)
        path = os.path.join(path, 'CelebA')
        self.split = split

        split_df = pd.read_csv(os.path.join(path, 'list_eval_partition.csv'))
        splits = {
            'train': 0,
            'valid': 1,
            'test': 2
        }
        partition_idx = split_df['split'] == splits[split]

        self.attr_df = pd.read_csv(os.path.join(path, 'list_attr_celeba.csv'), sep=' ').replace(-1, 0)
        self.attr_df = self.attr_df[partition_idx] # keep only relevant split train/val/test

        self.target = target
        self.path = path
        self.T = transform

    def download_dataset(self, path):
        url = "https://drive.google.com/uc?id=1ebDzE4vsjPB4klNyTywjrZqGhUsFxZqb"
        output = os.path.join(path, 'celeba.tar.gz')
        print(f'=> Downloading CelebA dataset from {url}')
        gdown.download(url, output, quiet=False)

        print('=> Extracting dataset..')
        tar = tarfile.open(os.path.join(path, 'celeba.tar.gz'), 'r:gz')
        tar.extractall(path=path)
        tar.close()
        os.remove(output)

    def __getitem__(self, index):
        data = self.attr_df.iloc[index]
        img_name = data['image']
        target_attr = data[self.target]

        image = PIL.Image.open(os.path.join(self.path, 'img_align_celeba', img_name))
        return self.T(image), target_attr 

    def __len__(self):
        return len(self.attr_df)