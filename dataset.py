import os
from random import randint

import cv2
import nibabel as nib
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import Dataset, SequentialSampler
from torchvision import datasets, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def init_dataset_loader(dataset, args, shuffle=False):
    _sampler = SequentialSampler(dataset)
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args['Batch_Size'], shuffle=shuffle,
                    num_workers=0, drop_last=True, pin_memory=True,
                    sampler=_sampler
                    )
            )

    return dataset_loader


# # apply albumentations on torch dataloader
# class Transforms:
#     def __init__(self, transforms: A.Compose):
#         self.transforms = transforms

#     def __call__(self, img, *args, **kwargs):
#         return self.transforms(image=np.array(img))["image"]

# custom_transform = A.Compose(
#     [
#         A.Resize(height=256, width=256),
#         # A.OneOf(
#         #     [A.Resize(height=224, width=224), A.RandomCrop(height=224, width=224)], p=1
#         # ),
#         # A.RandomCrop(height=224, width=224),
#         A.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),
#         A.OneOf(
#             [
#                 A.ElasticTransform(
#                     p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03
#                 ),
#                 A.GridDistortion(p=1),
#                 A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
#             ],
#             p=0.3,
#         ),
#         A.HorizontalFlip(p=0.5),
#         # A.Rotate(10),
#         # A.Normalize(mean=0.5, std=0.5),
#         ToTensorV2(),
#     ]
# )


class custom(Dataset):
    def __init__(self, dir, img_size=(256, 256), rgb=False):
        self.ROOT_DIR = dir

        transforms_list = [A.Resize(height=img_size[0], width=img_size[1]),
                           A.Normalize((0.5), (0.5)),
                           ToTensorV2()]

        if rgb:
            channels = 3
        else:
            channels = 1

        # print("\nAugmentation list\n", transforms_list)

        self.transform = A.Compose(transforms_list)

        self.rgb = rgb
        self.img_size = img_size
        self.classes = ["NORMAL","PNEUMONIA"]

        self.filenames = [f"{self.ROOT_DIR}/{i}" for i in os.listdir(self.ROOT_DIR)]

        for i in self.filenames[:]:
            if not ( i.endswith(".png") or i.endswith(".jpeg")):
                self.filenames.remove(i)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.rgb:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.filenames[idx]), 1), cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(os.path.join(self.filenames[idx]), 0)
        img = np.array(img).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img)   
            image = transformed['image']
        
        sample = {"image": image, "filenames": self.filenames[idx]}
            
        return sample


def load_CIFAR10(args, train=True):
    return torch.utils.data.DataLoader(
            datasets.CIFAR10(
                    "./DATASETS/CIFAR10", train=train, download=True, transform=transforms
                        .Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                                ]
                            )
                    ),
            shuffle=True, batch_size=args["Batch_Size"], drop_last=True
            )

