from fix_match import FixMatch, EMA, resNet
import torch
import time
import torch.nn.functional as F
from data_loader import load_data
import torch.optim as optim
from config import train_path, test_path, val_path, unlabel_path, lambda_u, num_class,\
    mu, batch_size, lr, beta, weight_decay, threshold, warmup, total_steps, eval_step, ema_decay
from tqdm import tqdm
from config import device
from torch.optim.lr_scheduler import LambdaLR
import math
from os.path import join as pjoin
import logging
import os
import pandas as pd
import numpy as np
import random

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

logger = None


def init_log():
    """ 
    initlizate log
    """
    global logger
    logger = logging.getLogger('FixMatch')
    logger.setLevel(logging.INFO)
    log_path = pjoin('out', 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = pjoin(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def run_batch(label_img, label, weak_img, strong_img, model, lambda_u, threshold):
    weak_img = weak_img.to(device)
    strong_img = strong_img.to(device)
    label_img = label_img.to(device)
    label = label.to(device)
    out, a_u, A_u = FixMatch.forward(model, label_img, weak_img, strong_img)
    acc = (torch.argmax(out, dim=1) == label).sum().item() / len(label)
    # 1) Cross-entropy loss for labeled data.
    l_x = F.cross_entropy(out, label, reduction='mean')

    # 2) Cross-entropy loss with pseudo-label B and conﬁdence for unlabeled data
    max_probs, a_u_label = torch.max(F.softmax(a_u.detach()), dim=-1)
    mask = max_probs.ge(threshold).float()
    l_u = (F.cross_entropy(A_u, a_u_label, reduction='none') * mask).mean()
    loss = l_x + lambda_u * l_u
    return loss, acc


def run_val_epoch(model, val_loader):

    loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for img, label in val_loader:
            img = img.to(device)
            label = label.to(device)
            out = FixMatch.predict(model, img)
            acc += (torch.argmax(out, dim=1) ==
                    label).sum().item() / len(label)
            L = F.cross_entropy(out, label, reduction='mean')
            loss += L.item()
    return loss / len(val_loader), acc / len(val_loader)


def run_train_epoch(model, op, train_loader, unlabel_loader,
                    max_batch, lambda_u, threshold, scheduler,
                    ema_model):
    model.train()
    loss = 0.0
    total_acc = 0.0
    labeled_iter = iter(train_loader)
    unlabeled_iter = iter(unlabel_loader)

    for _ in range(max_batch):
        try:
            img, label = labeled_iter.next()
        except Exception:
            labeled_iter = iter(train_loader)
            img, label = labeled_iter.next()
            logger.info('reload dataset')

        try:
            weak_img, strong_img = unlabeled_iter.next()
        except Exception:
            unlabeled_iter = iter(unlabel_loader)
            weak_img, strong_img = unlabeled_iter.next()
            logger.info('reload dataset')

        L, acc = run_batch(img, label, weak_img, strong_img,
                           model, lambda_u, threshold)
        total_acc += acc
        loss += L.item()
        L.backward()
        op.step()
        scheduler.step()
        ema_model.update(model)
        model.zero_grad()

    return loss / max_batch, total_acc / max_batch


def save_checkpoint(check_point, is_best, path):
    torch.save(check_point, f'{path}/checkpoint.pt')
    if is_best:
        torch.save(check_point, f'{path}/best_checkpoint.pt')


def train(model, epochs, ema_model, op, scheduler, train_loader, val_loader,
          unlabel_loader, epoch=0, best_acc=0.0, path=''):
    val_loss_list = []
    train_loss_list = []
    val_acc_list = []
    train_acc_list = []
    for epoch in tqdm(range(epoch, epochs, 1)):
        start = time.time()
        train_loss, train_acc = run_train_epoch(model, op, train_loader, unlabel_loader,
                                                eval_step, lambda_u, threshold,
                                                scheduler, ema_model)
        val_loss, val_acc = run_val_epoch(ema_model.ema, val_loader)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)
        interval = time.time() - start
        logger.info(
            f'[Epoch]:{epoch+1}/{epochs}, train_loss:{train_loss}, train_acc: {train_acc}, val_loss:{val_loss}, val_acc:{val_acc}, time:{interval}s')
        is_best = False
        if best_acc < val_acc:
            best_acc = val_acc
            is_best = True
        ema_save = ema_model.ema.module if hasattr(
            ema_model.ema, "module") else ema_model.ema
        check_point = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'ema_state_dict': ema_save.state_dict(),
            'optimizer': op.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_acc': best_acc
        }
        save_checkpoint(check_point, is_best, path)


def predict(model, test_loader, labels, test_files):
    predicts = []
    for img in test_loader:
        img = img.to(device)
        out = FixMatch.predict(model, img)
        predict_labels = torch.argmax(out, dim=1).tolist()
        predicts.extend([labels[x] for x in predict_labels])
    result = pd.DataFrame({
        'image': test_files,
        'class': predicts,
    })
    result.to_csv('predict.txt', index=False, header=False)


def load_model(path, model, op, scheduler, ema_model):
    checkpoint = torch.load(f'{path}/best_checkpoint.pt')
    best_acc = checkpoint['best_acc']
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    op.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, op, scheduler, ema_model, epoch, best_acc


def main(train_path, val_path, test_path, unlabel_path, test=True, resume=True, path='',net_size = 152):
    set_seed()
    init_log()
    logger.info("Starting train model")
    model = resNet(num_class, net_size)
    model.to(device)
    ema_model = EMA(device, model, ema_decay)
    op = optim.SGD(model.parameters(), lr=lr,
                   weight_decay=weight_decay, momentum=beta, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(
        op, warmup, total_steps)
    if resume:
        model, op, scheduler, ema_model, epoch, best_acc = load_model(
            path, model, op, scheduler, ema_model)
    else:
        epoch, best_acc = 0, 0
    logger.info('handle dataset')
    train_loader, val_loader, test_loader, unlabel_loader, labels, test_files = load_data(
        train_path, val_path, test_path, unlabel_path, batch_size, mu)
    epochs = math.ceil(total_steps / eval_step)
    if test:
        predict(model, test_loader, labels, test_files)
    else:
        train(model, epochs, ema_model, op, scheduler,
              train_loader, val_loader, unlabel_loader,
              epoch, best_acc, path)


if __name__ == "__main__":
    main(train_path, val_path, test_path,
         unlabel_path, test=False, resume=False, path='')
