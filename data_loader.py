from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from config import train_path, val_path, test_path, unlabel_path, batch_size, mu
from PIL import Image
import numpy as np
import torch


def get_mean_and_std(loader):

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader)
    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0, 2])

    std = torch.sqrt(var / (len(loader) * 224 * 224))
    return mean, std


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


def normalize_images(images, normalization):
    new_images = []
    for image in images:
        new_images.append(normalization(image))
    return new_images


def load_val_images(path, labels, normalization):
    txt_path = path + "/val_annotations.txt"
    val_data = []
    with open(txt_path, 'r') as f:
        for line in tqdm(f.readlines()):
            words = line.split('\t')
            name, label = words[0], words[1]
            img = np.array(Image.open(f'{path}/images/{name}').convert('RGB'))
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = normalization(img)
            val_data.append([img, labels[label]])
    return val_data


def load_unlabeled_data(path, normalization):
    data = []
    images = os.listdir(path)
    for image in tqdm(images):
        img = np.array(Image.open(f'{path}/{image}').convert('RGB'))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = normalization(img)
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
    mean, std = get_mean_and_std(train_loader)
    print(mean, std)
    normalization = transforms.Normalize(mean, std)
    train_dataset = normalize_images(train_dataset, normalization)
    train_loader = DataLoader(LabelDataSet(train_dataset), batch_size=batch_size,
                              shuffle=True)
    val_dataset = load_val_images(val_path, labels, normalization)
    val_loader = DataLoader(LabelDataSet(val_dataset), batch_size=batch_size,
                            shuffle=True)
    test_dataset = load_unlabeled_data(test_path, normalization)
    test_loader = DataLoader(UnLabelDataSet(test_dataset, test=True), batch_size=batch_size,
                             shuffle=True)
    unlabel_dataset = load_unlabeled_data(unlabel_path, normalization)
    unlabel_loader = DataLoader(UnLabelDataSet(unlabel_dataset), batch_size=mu * batch_size,
                                shuffle=True)
    return train_loader, val_loader, test_loader, unlabel_loader, labels


if __name__ == "__main__":
    train_loader, val_loader, test_loader, unlabel_loader, labels = load_data(
        train_path, val_path, test_path, unlabel_path, batch_size, mu)
    iter(unlabel_loader).next()
