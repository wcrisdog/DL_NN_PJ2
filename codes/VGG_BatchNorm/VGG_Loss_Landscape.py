import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader


device = torch.device('cpu')

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            loss_list.append(loss.item())


            loss.backward()
            grad_norm = model.classifier[-1].weight.grad.norm().item()
            grad.append(grad_norm)
            optimizer.step()

        train_accuracy_curve[epoch] = get_accuracy(model, train_loader)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader)
        learning_curve[epoch] = sum(loss_list) / len(loss_list)
        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        axes[0].plot(learning_curve)

        model.eval()
        # after computing train/val accuracy:
        axes[1].plot(val_accuracy_curve, label='val acc')
        axes[1].plot(train_accuracy_curve, label='train acc')
        axes[1].legend()

    return learning_curve, train_accuracy_curve, val_accuracy_curve, losses_list, grads

if '__main__' == __name__:
    # Train your model
    # feel free to modify
    epo = 20
    loss_save_path = ''
    grad_save_path = ''

    set_random_seeds(seed_value=2025, device=device)
    model = VGG_A()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    curves_base = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    _, _, _, loss, grads = curves_base
    np.savetxt(os.path.join(loss_save_path, 'loss_base.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, 'grads_base.txt'), grads, fmt='%s', delimiter=' ')

    set_random_seeds(seed_value=2025, device=device)
    bn_model = VGG_A_BatchNorm()
    lr = 0.001
    optimizer = torch.optim.Adam(bn_model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    curves_bn = train(bn_model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    _, _, _, loss, grads = curves_bn
    np.savetxt(os.path.join(loss_save_path, 'loss_bn.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(grad_save_path, 'grads_bn.txt'), grads, fmt='%s', delimiter=' ')

    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    min_curve = []
    max_curve = []

    # Use this function to plot the final loss landscape,
    # fill the area between the two curves can use plt.fill_between()
    def plot_loss_landscape(curves_base, curves_bn):
        epochs = range(len(curves_base[0]))
        # Unpack
        loss_base, train_acc_base, val_acc_base, *_ = curves_base
        loss_bn, train_acc_bn, val_acc_bn, *_ = curves_bn

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_base, label='VGG-A')
        plt.plot(epochs, loss_bn, label='VGG-A + BN')
        plt.xlabel('Epoch')
        plt.ylabel('Avg. training loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('loss_curve.png')

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc_base, label='Train acc (no BN)')
        plt.plot(epochs, train_acc_bn, label='Train acc (BN)')
        plt.plot(epochs, val_acc_base, '--', label='Val acc (no BN)')
        plt.plot(epochs, val_acc_bn, '--', label='Val acc (BN)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_curve.png')

    plot_loss_landscape(curves_base, curves_bn)