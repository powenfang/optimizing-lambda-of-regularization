import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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