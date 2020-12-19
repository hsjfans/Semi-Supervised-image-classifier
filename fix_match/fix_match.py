from fix_match.resnet_18 import ResNet18
import torch.nn as nn
import torch
import torch.nn.functional as F


class FixMatch(nn.Module):

    def __init__(self, num_classes):
        super(FixMatch, self).__init__()
        self.net = ResNet18(num_classes)

    def predict(self, img):
        return F.softmax(self.net(img))

    def forward(self, label_img, weak_img, strong_img):
        # print(label_img.shape, weak_img.shape, strong_img.shape)
        label_size, aug_size = label_img.shape[0], label_img.shape[0]
        x = torch.cat([label_img, weak_img, strong_img], dim=0)
        out = F.softmax(self.net(x))
        label_out, a_u, A_u = torch.split(
            out, [label_size, aug_size, aug_size], dim=0)
        return label_out, a_u, A_u
