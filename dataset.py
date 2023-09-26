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


# helper function to make getting another batch of data easier


# from diffusion_training import output_img


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def init_datasets(ROOT_DIR, args):
    training_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}train/NORMAL/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    testing_dataset = MRIDataset(
            ROOT_DIR=f'{ROOT_DIR}test/NORMAL/', img_size=args['img_size'], random_slice=args['random_slice']
            )
    return training_dataset, testing_dataset


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

def init_data4eval(dir, args):
    # training_dataset = MVTec(
    #                 dir = "../data/chest_xray/train/", anomalous=False, img_size=args["img_size"],
    #                 rgb=True
    #                 )
    testing_dataset = MVTec(
            dir = "../data/chest_xray/test/", anomalous=False, img_size=args["img_size"],
            rgb=True, include_good=True
            )
    print(len(testing_dataset))
    # training_dataset_loader = init_dataset_loader(training_dataset, args)
    testing_dataset_loader = init_dataset_loader(testing_dataset, args)

    return testing_dataset_loader


class MVTec(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=True, random_crop=True, include_good=False):
        # dir = './DATASETS/leather'

        self.ROOT_DIR = dir
        self.anomalous = anomalous
        if not anomalous:
            self.ROOT_DIR += "/train/good"

        transforms_list = [transforms.ToPILImage()]

        if rgb:
            channels = 3
        else:
            channels = 1
            transforms_list.append(transforms.Grayscale(num_output_channels=channels))
        transforms_mask_list = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=channels)]
        if not random_crop:
            transforms_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
            transforms_mask_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
        transforms_list.append(transforms.ToTensor())
        transforms_mask_list.append(transforms.ToTensor())
        if rgb:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transforms_list.append(transforms.Normalize((0.5), (0.5)))
        transforms_mask_list.append(transforms.Normalize((0.5), (0.5)))
        self.transform = transforms.Compose(transforms_list)
        self.transform_mask = transforms.Compose(transforms_mask_list)

        self.rgb = rgb
        self.img_size = img_size
        self.random_crop = random_crop
        self.classes = ["color", "cut", "fold", "glue", "poke"]
        if include_good:
            self.classes.append("good")
        if anomalous:
            self.filenames = [f"{self.ROOT_DIR}/test/{i}/{x}" for i in self.classes for x in
                              os.listdir(self.ROOT_DIR + f"/test/{i}")]

        else:
            self.filenames = [f"{self.ROOT_DIR}/{i}" for i in os.listdir(self.ROOT_DIR)]

        for i in self.filenames[:]:
            if not i.endswith(".png"):
                self.filenames.remove(i)
        self.filenames = sorted(self.filenames, key=lambda x: int(x[-7:-4]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"filenames": self.filenames[idx]}
        if self.rgb:
            sample["image"] = cv2.cvtColor(cv2.imread(os.path.join(self.filenames[idx]), 1), cv2.COLOR_BGR2RGB)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            sample["image"] = cv2.imread(os.path.join(self.filenames[idx]), 0)
            sample["image"] = sample["image"].reshape(*sample["image"].shape, 1)

        if self.anomalous:
            file = self.filenames[idx].split("/")
            if file[-2] == "good":
                sample["mask"] = np.zeros((sample["image"].shape[0], sample["image"].shape[1], 1)).astype(np.uint8)
            else:
                sample["mask"] = cv2.imread(
                        os.path.join(self.ROOT_DIR, "ground_truth", file[-2], file[-1][:-4] + "_mask.png"), 0
                        )
        if self.random_crop:
            x1 = randint(0, sample["image"].shape[-2] - self.img_size[1])
            y1 = randint(0, sample["image"].shape[-3] - self.img_size[0])
            if self.anomalous:
                sample["mask"] = sample["mask"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]
            sample["image"] = sample["image"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]

        if self.transform:
            sample["image"] = self.transform(sample["image"])
            if self.anomalous:
                sample["mask"] = self.transform_mask(sample["mask"])
                sample["mask"] = (sample["mask"] > 0).float()

        return sample



# apply albumentations on torch dataloader
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]

custom_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        # A.OneOf(
        #     [A.Resize(height=224, width=224), A.RandomCrop(height=224, width=224)], p=1
        # ),
        # A.RandomCrop(height=224, width=224),
        A.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),
        A.OneOf(
            [
                A.ElasticTransform(
                    p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ],
            p=0.3,
        ),
        A.HorizontalFlip(p=0.5),
        # A.Rotate(10),
        # A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ]
)


class custom(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=False, e_aug=False):
        # dir = './DATASETS/leather'

        self.ROOT_DIR = dir
        self.anomalous = anomalous
        # if not anomalous:
        #     self.ROOT_DIR += "/NORMAL/"

        # transforms_list = [transforms.ToPILImage(), transforms.RandomAffine(3, translate=(0.02, 0.09))]
        transforms_list = []

        if rgb:
            channels = 3
        else:
            channels = 1
            # transforms_list.append(A.ToGray(always_apply=True, p=1))
        # transforms_mask_list = [A.ToGray()]

        if e_aug:
            transforms_list.extend(
                [
                    A.Resize(height=img_size[0], width=img_size[1]),
                    A.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),
                    A.OneOf(
                        [
                            A.ElasticTransform(
                                p=1, alpha=60, sigma=120 * 0.05, alpha_affine=120 * 0.03
                            ),
                            A.GridDistortion(p=1),
                            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                        ],
                        p=1,
                    ),
                    # A.HorizontalFlip(p=0.5)
                ]
        )
        else:
            transforms_list.extend(
                [
                    A.Resize(height=img_size[0], width=img_size[1]),
                    # A.Affine(rotate=(-10, 10), scale=(0.8, 1.2)),
                    # A.HorizontalFlip(p=0.5)
                ]
        )
            # transforms_mask_list.append(A.Resize(height=img_size, width=img_size))
        transforms_list.append(A.Normalize((0.5), (0.5)))
        transforms_list.append(ToTensorV2())
        # transforms_mask_list.append(ToTensorV2())

        # if rgb:
        #     transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        # else:
        #     transforms_list.append(transforms.Normalize((0.5), (0.5)))
        # transforms_mask_list.append(transforms.Normalize((0.5), (0.5)))
        # self.transform = Transforms(A.Compose(transforms_list))
        print(transforms_list)

        self.transform = A.Compose(transforms_list)

        # self.transform_mask = Transforms(A.Compose(transforms_mask_list))


        self.rgb = rgb
        self.img_size = img_size
        self.classes = ["NORMAL","PNEUMONIA"]
        # if include_good:
            # self.classes.append("good")
        if anomalous:
            self.filenames = [f"{self.ROOT_DIR}/test/{i}/{x}" for i in self.classes for x in
                              os.listdir(self.ROOT_DIR + f"/test/{i}")]

        else:
            self.filenames = [f"{self.ROOT_DIR}/{i}" for i in os.listdir(self.ROOT_DIR)]
            # print(self.filenames)

        for i in self.filenames[:]:
            if not ( i.endswith(".png") or i.endswith(".jpeg")):
                self.filenames.remove(i)
        # self.filenames = sorted(self.filenames, key=lambda x: int(x[-7:-4]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.rgb:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.filenames[idx]), 1), cv2.COLOR_BGR2RGB)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            img = cv2.imread(os.path.join(self.filenames[idx]), 0)
            # sample["image"] = sample["image"].reshape(*sample["image"].shape, 1)
        img = np.array(img).astype(np.float32)
        # print(img.shape)
        # print(img.dtype)

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

