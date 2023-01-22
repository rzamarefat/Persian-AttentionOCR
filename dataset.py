import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from tokenizer import Tokenizer
from glob import glob 


class CustomDataset(Dataset):
    def __init__(self, 
                root_path_to_images,
                path_to_gt_file,
                chars
                img_width, 
                img_height, 
                is_train=True
                ):
        self.img_dim = (img_width, img_height)

        self.is_train = is_train
        self.path_to_gt_file = path_to_gt_file
        
        self.chars = chars
        self.tokenizer = Tokenizer(self.chars)

        with open(path_to_gt_file) as h:
            content = [l.replace("\n", "") for l in h.readlines()]

        self.gt_data = {}
        for c in content:
            self.gt_data[c.split(" --> ")[0].split("/")[-1]] = c.split(" --> ")[1]

        self.max_length_words = 0
        for img_p, gt in self.gt_data.items():
            if len(gt) > self.max_length_words:
                self.max_length_words = len(gt)

        self.text_images = [f for f in sorted(glob(os.path.join(self.root_path_to_images, "*")))]


        self.img_trans = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225)),
                        ])


    def __len__(self):
        return len(self.text_images)

    def __getitem__(self, index):

        target_img = self.text_images[index]
        gt = self.gt_data[target_img.split("/")[-1]]
        img = Image.open(target_img)
        img = img.convert("RGB")
        img = img.resize((160, 60))
        img = self.img_trans(img)


        label = torch.full((self.max_length_words + 2, ), self.tokenizer.EOS_token, dtype=torch.long)

        ts = self.tokenizer.tokenize(gt)
        label[:ts.shape[0]] = torch.tensor(ts)
        
        return img, label


if __name__ == "__main__":
    img_width = 160
    img_height = 60
    ds = CustomDataset(img_width, img_height, 10000, 4)

    for img, label in ds:
        print(img.shape)
        print(label)
        exit()