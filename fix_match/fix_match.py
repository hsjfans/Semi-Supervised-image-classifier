import torch


class FixMatch():

    @staticmethod
    def interleave(x, size):
        s = list(x.shape)
        return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def de_interleave(x, size):
        s = list(x.shape)
        return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

    @staticmethod
    def predict(net, img):
        return net(img)

    @staticmethod
    def forward(net, label_img, weak_img, strong_img):
        # print(label_img.shape, weak_img.shape, strong_img.shape)
        label_size, aug_size = label_img.shape[0], weak_img.shape[0]
        mu = aug_size // label_size
        x = FixMatch.interleave(
            torch.cat([label_img, weak_img, strong_img], dim=0), 2 * mu + 1)
        out = net(x)
        out = FixMatch.de_interleave(out, 2 * mu + 1)
        label_out, a_u, A_u = torch.split(
            out, [label_size, aug_size, aug_size], dim=0)
        return label_out, a_u, A_u

    # def __call__(self, net, label_img, weak_img, strong_img):
    #     return self.forward(net, label_img, weak_img, strong_img)
