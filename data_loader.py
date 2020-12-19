from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from config import train_path, val_path, test_path, unlabel_path, batch_size, mu
from PIL import Image
import numpy as np
import torch


def weak_transform(img):
    weak_img = transforms.RandomHorizontalFlip(p=0.5)(img)
    return weak_img


def strong_transform(img):

    # Fill cutout with gray color
    gray_code = 127

    # ratio=(1, 1) to set aspect ratio of square
    # p=1 means probability is 1, so always apply cutout
    # scale=(0.01, 0.01) means we want to get cutout of 1% of image area
    # Hence: Cuts out gray square of 52*52
    cutout_img = transforms.RandomErasing(p=1,
                                          ratio=(1, 1),
                                          scale=(0.01, 0.01),
                                          value=gray_code)(img)
    return cutout_img


def load_train_images(path):
    # 暂时先不使用box来测试
    img_classes = os.listdir(path)
    data = []
    labels = {}
    for i, image_class in tqdm(enumerate(img_classes)):
        labels[image_class] = i
        images = os.listdir(f'{path}/{image_class}/images')
        for image in images:
            img = torch.from_numpy(np.array(Image.open(
                f'{path}/{image_class}/images/{image}').convert('RGB'))).permute(2, 0, 1).float()
            data.append([img, i])
    return data, labels


def load_val_images(path, labels):
    txt_path = path + "/val_annotations.txt"
    val_data = []
    with open(txt_path, 'r') as f:
        for line in tqdm(f.readlines()):
            words = line.split('\t')
            name, label = words[0], words[1]
            img = torch.from_numpy(
                np.array(Image.open(f'{path}/images/{name}').convert('RGB'))).permute(2, 0, 1).float()
            val_data.append([img, labels[label]])
    return val_data


def load_unlabeled_data(path):
    data = []
    images = os.listdir(path)
    for image in tqdm(images):
        img = torch.from_numpy(
            np.array(Image.open(f'{path}/{image}').convert('RGB'))).permute(2, 0, 1).float()
        data.append(img)
    return data


class LabelDataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


class UnLabelDataSet(Dataset):

    def __init__(self, data, test=False):
        self.data = data
        self.test = test

    def __getitem__(self, index: int):
        img = self.data[index]
        if self.test:
            return img
        else:
            return weak_transform(img), strong_transform(img)

    def __len__(self):
        return len(self.data)


def load_data(train_path, val_path, test_path, unlabel_path, batch_size, mu):
    train_dataset, labels = load_train_images(train_path)
    train_loader = DataLoader(LabelDataSet(train_dataset), batch_size=batch_size,
                              shuffle=True)
    val_dataset = load_val_images(val_path, labels)
    val_loader = DataLoader(LabelDataSet(val_dataset), batch_size=batch_size,
                            shuffle=True)
    test_dataset = load_unlabeled_data(test_path)
    test_loader = DataLoader(UnLabelDataSet(test_dataset, test=True), batch_size=batch_size,
                             shuffle=True)
    unlabel_dataset = load_unlabeled_data(unlabel_path)
    unlabel_loader = DataLoader(UnLabelDataSet(unlabel_dataset), batch_size=mu * batch_size,
                                shuffle=True)
    return train_loader, val_loader, test_loader, unlabel_loader, labels


if __name__ == "__main__":
    train_loader, val_loader, test_loader, unlabel_loader, labels = load_data(
        train_path, val_path, test_path, unlabel_path, batch_size, mu)
    iter(unlabel_loader).next()
