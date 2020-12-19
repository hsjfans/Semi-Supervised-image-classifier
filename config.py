import torch
train_path = '/Users/mitmit/Desktop/images_data/train'
val_path = '/Users/mitmit/Desktop/images_data/val'
test_path = '/Users/mitmit/Desktop/images_data/test/images'
unlabel_path = '/Users/mitmit/Desktop/images_data/unlabel_from_train'

# The hyper
num_class = 200
lambda_u = 1
mu = 1
batch_size = 32
lr = 0.03
beta = 0.9
weight_decay = 0.001
epochs = 10
num_classes = 200
threshold = 0.95


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
