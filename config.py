import torch
train_path = '/Users/mitmit/Desktop/images_data/train'
val_path = '/Users/mitmit/Desktop/images_data/val'
test_path = '/Users/mitmit/Desktop/images_data/test/images'
unlabel_path = '/Users/mitmit/Desktop/images_data/unlabel_from_train'

# The hyper
num_class = 200
lambda_u = 1
mu = 2
batch_size = 32
lr = 0.03
beta = 0.8
weight_decay = 0.001
num_classes = 200
threshold = 0.95
warmup = 0
total_steps = 2**16
eval_step = 2**6
ema_decay = 0.999

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
