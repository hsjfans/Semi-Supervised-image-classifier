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
        except:
            labeled_iter = iter(train_loader)
            img, label = labeled_iter.next()

        try:
            weak_img, strong_img = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabel_loader)
            weak_img, strong_img = unlabeled_iter.next()

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


def save_model(model, epoch):
    check_point = {'model': model.state_dict(), 'epoch': epoch}
    torch.save(check_point, f'epoch_{epoch}.pt')


if __name__ == "__main__":

    model = resNet(num_class, 34)
    model.to(device)
    op = optim.SGD(model.parameters(), lr=lr,
                   weight_decay=weight_decay, momentum=beta, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(
        op, warmup, total_steps)
    train_loader, val_loader, test_loader, unlabel_loader, labels = load_data(
        train_path, val_path, test_path, unlabel_path, batch_size, mu)
    epochs = math.ceil(total_steps / eval_step)
    ema_model = EMA(device, model, ema_decay)

    val_loss_list = []
    train_loss_list = []
    val_acc_list = []
    train_acc_list = []
    for epoch in tqdm(range(epochs)):
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
        print(f'[Epoch]:{epoch}/{epochs}, train_loss:{train_loss}, train_acc: {train_acc}, val_loss:{val_loss}, val_acc:{val_acc}, time:{interval}s')
        if epoch % 5 == 0:
            save_model(model, epoch)
    save_model(model, epochs)
