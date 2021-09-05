import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

def form_test_name(batch_norm, auto_l2, adv_l2, ts):
    if batch_norm:
        if auto_l2:
            test_name = '_'.join(['bn', 'autol2', ts])
        elif adv_l2:
            test_name = '_'.join(['bn', 'advl2', ts])
        else:
            test_name = '_'.join(['bn', 'l2', ts])
    else:
        if auto_l2:
            test_name = '_'.join(['autol2', ts])
        elif adv_l2:
            test_name = '_'.join(['advl2', ts])
        else:
            test_name = '_'.join(['l2', ts])
    return test_name

def get_train_acc(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    n_correct = (predicted == labels).float().sum().item()
    n_samples = labels.size(0)
    train_acc = 100.0 * n_correct / n_samples
    return train_acc

def loss_or_error_increase(loss, acc, min_loss, max_acc):
    if loss >= min_loss:
        return True
    if acc <= max_acc:
        return True
    return False

# when loss or accuracy increase by pctg %
def loss_or_error_increase_by_percentage(loss, acc, min_loss, max_acc, pct):
    if loss >= min_loss * (1 + pct/100):
        return True
    if acc * (1 + pct/100) <= max_acc:
        return True
    return False


def L2_penalty(params, Lambda_L2):
    L2_norm = sum(p.pow(2.0).sum() for p in params)
    return Lambda_L2 * L2_norm

def get_test_result(model, test_loader, Lambda_L2, device):
    
    test_bareloss, test_loss = 0, 0
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)

            bareloss = F.cross_entropy(outputs, labels)
            L2 = L2_penalty(model.parameters(), Lambda_L2)
            loss = bareloss + L2

            test_bareloss += bareloss.item()
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        test_acc = 100.0 * n_correct / n_samples
    
    return test_bareloss / n_samples, test_loss / n_samples, test_acc 