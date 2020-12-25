from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from config import train_path, val_path, test_path, unlabel_path, batch_size, mu
from PIL import Image
import torch
from randaugment import RandAugment

normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_mean_and_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, _ in loader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(loader))
    std.div_(len(loader))
    return mean, std


class UnLabelTransform:
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, img):
        weak = self.weak(img)
        strong = self.strong(img)
        return self.normalize(weak), self.normalize(strong)


def load_train_images(path):
    # 暂时先不使用box来测试
    img_classes = os.listdir(path)
    data = []
    labels = {}
    for i, image_class in tqdm(enumerate(img_classes)):
        labels[image_class] = i
        images = os.listdir(f'{path}/{image_class}/images')
        for image in images:
            img = Image.open(
                f'{path}/{image_class}/images/{image}')
            #  torch.from_numpy().permute(2, 0, 1).float()
            data.append([img, i])
    return data, labels


def load_val_images(path, labels):
    txt_path = path + "/val_annotations.txt"
    val_data = []
    with open(txt_path, 'r') as f:
        for line in tqdm(f.readlines()):
            words = line.split('\t')
            name, label = words[0], words[1]
            img = Image.open(f'{path}/images/{name}')
            # img = torch.from_numpy(img).permute(2, 0, 1).float()
            # img = normalization(img)
            val_data.append([img, labels[label]])
    return val_data


def load_unlabeled_data(path):
    data = []
    images = os.listdir(path)
    for image in tqdm(images):
        img = Image.open(f'{path}/{image}')
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        # img = normalization(img)
        data.append(img)
    return data


class LabelDataSet(Dataset):

    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=normal_mean, std=normal_std)
        ])

    def __getitem__(self, index: int):
        return self.transform(self.data[index][0]), self.data[index][1]

    def __len__(self):
        return len(self.data)


class UnLabelDataSet(Dataset):

    def __init__(self, data, test=False):
        self.data = data
        self.test = test
        self.transform = UnLabelTransform(normal_mean, normal_std)
        self.labeled_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=normal_mean, std=normal_std)
        ])

    def __getitem__(self, index: int):
        img = self.data[index]
        if self.test:
            return self.labeled_transform(img)
        else:
            return self.transform(img)

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

    # img = Image.open(f'{test_path}/test_0.JPEG')
    # # img = np.array(img.convert('RGB'))
    # weak = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(size=64,
    #                           padding=int(64 * 0.125),
    #                           padding_mode='reflect'),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=normal_mean, std=normal_std)
    # ])

    # weak_img = weak(img)
    # print(weak_img.shape)
