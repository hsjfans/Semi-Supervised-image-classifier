"""
Analysis the runtime log .
"""
import matplotlib.pyplot as plt
# import numpy as np
from tqdm import tqdm


def extract_info(log_path='./out/logs/train.log'):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    epochs = 0
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.find('train_acc') != -1:
                epochs += 1
                words = line.split(',')
                train_loss.append(float(words[2].split(':')[1]))
                train_acc.append(float(words[3].split(':')[1]))
                val_loss.append(float(words[4].split(':')[1]))
                val_acc.append(float(words[5].split(':')[1]))
    # print(train_loss)
    return [epoch for epoch in range(epochs)], train_acc, train_loss, val_acc, val_loss


def draw(epochs, train_infos, val_infos, name='acc'):
    plt.plot(epochs, train_infos, label=f'train_{name}')
    plt.plot(epochs, val_infos, label=f'val_{name}')
    plt.xlabel('epoch')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.show()


def analysis_infos(epochs, train_acc, train_loss, val_acc, val_loss):
    draw(epochs, train_loss, val_loss, name='loss')
    draw(epochs, train_acc, val_acc)


if __name__ == "__main__":
    epochs, train_acc, train_loss, val_acc, val_loss = extract_info()
    analysis_infos(epochs, train_acc, train_loss, val_acc, val_loss)
