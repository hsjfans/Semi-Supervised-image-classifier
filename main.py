from fix_match import FixMatch
import torch
import torch.nn.functional as F
from data_loader import load_data
import torch.optim as optim
from config import train_path, test_path, val_path, unlabel_path, lambda_u, num_class,\
    mu, batch_size, lr, beta, weight_decay, epochs, threshold
from tqdm import tqdm
#  train_loader, test_dataset, val_loader, unlabel_loader


def run_batch(label_img, label, weak_img, strong_img, model, lambda_u, threshold):

    out, a_u, A_u = model(label_img, weak_img, strong_img)
    # 1) Cross-entropy loss for labeled data.
    l_x = F.cross_entropy(out, label)

    # 2) Cross-entropy loss with pseudo-label B and conﬁdence for unlabeled data
    max_probs, _ = torch.max(a_u, dim=1)
    mask = max_probs.ge(threshold).float()
    l_u = (F.cross_entropy(A_u, torch.argmax(
        a_u, dim=1), reduction='none') * mask).mean()
    loss = l_x + lambda_u * l_u
    return loss


def run_val_epoch(model, val_loader, batch_size):
    model.eval()
    loss = 0.0
    acc = 0.0
    for img, label in val_loader:
        out = model.predict(img)
        acc += (torch.argmax(out, dim=1) == label).sum().item() / batch_size
        L = F.cross_entropy(out, label)
        loss += L.item()
    return loss / len(val_loader), acc / len(val_loader)


def run_train_epoch(model, op, train_loader, unlabel_loader, max_batch, lambda_u, threshold):
    model.train()
    loss = 0.0
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

        L = run_batch(img, label, weak_img, strong_img,
                      model, lambda_u, threshold)
        loss += L.item()
        L.backward()
        op.step()

    return loss / max_batch


def save_model(model, epoch):
    check_point = {'model': model.state_dict(), 'epoch': epoch}
    torch.save(check_point, f'epoch_{epoch}.pt')


if __name__ == "__main__":

    model = FixMatch(num_class)
    op = optim.SGD(model.parameters(), lr=lr,
                   weight_decay=weight_decay, momentum=beta)
    train_loader, val_loader, test_loader, unlabel_loader, labels = load_data(
        train_path, val_path, test_path, unlabel_path, batch_size, mu)
    max_batch = int(max(len(train_loader) / batch_size,
                        len(unlabel_loader) / batch_size / mu))
    val_loss_list = []
    train_loss_list = []
    val_acc_list = []
    for epoch in tqdm(range(epochs)):
        train_loss = run_train_epoch(model, op, train_loader, unlabel_loader,
                                     max_batch, lambda_u, threshold)
        val_loss, val_acc = run_val_epoch(model, val_loader, batch_size)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(
            f'[Epoch]:{epoch}/{epochs}, train_loss:{train_loss}, val_loss:{val_loss}, val_acc:{val_acc}')
        if epoch % 2 == 0:
            save_model(model, epoch)
    save_model(model, epochs)
