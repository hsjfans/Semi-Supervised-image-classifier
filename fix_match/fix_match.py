import torch
from fix_match.resnet_18 import resNet


class FixMatch():

    def __init__(self, num_class, ty=34):
        super(FixMatch, self).__init__()
        self.net = resNet(num_class, ty=ty)

    def predict(self, img):
        return self.net(img)

    def forward(self, label_img, weak_img, strong_img):
        # print(label_img.shape, weak_img.shape, strong_img.shape)
        label_size, aug_size = label_img.shape[0], weak_img.shape[0]
        x = torch.cat([label_img, weak_img, strong_img], dim=0)
        out = self.net(x)
        label_out, a_u, A_u = torch.split(
            out, [label_size, aug_size, aug_size], dim=0)
        return label_out, a_u, A_u

    def __call__(self, label_img, weak_img, strong_img):
        return self.forward(label_img, weak_img, strong_img)

    def parameters(self):
        return self.net.parameters()

    def to(self, device):
        self.net.to(device)

    def train(self):
        return self.net.train()

    def eval(self):
        return self.net.eval()

    def state_dict(self):
        return self.net.state_dict()
