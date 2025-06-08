import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CIFARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, downsample=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        if downsample or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        loss = logits[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

def main():
    # Data loading & augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Model, loss, optimizer, scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = CIFARNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Histories
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    # Training & validation loops with checkpoint saving
    for epoch in range(1, 2):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_loss = running_loss/total
        train_acc = correct/total
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Acc {100.*train_acc:.2f}%")
        if epoch % 2 == 0:
            torch.save(net.state_dict(), f"cifar_net_epoch{epoch}.pth")

        # Validation
        net.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()*inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_loss = running_loss/total
        val_acc = correct/total
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        print(f"Validation Loss {val_loss:.4f}, Acc {100.*val_acc:.2f}%")
        scheduler.step()

    # Plot training curves
    plt.figure()
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.show()
    plt.figure()
    plt.plot([a*100 for a in train_acc_hist], label='Train Acc')
    plt.plot([a*100 for a in val_acc_hist], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.show()

    # Visualize Conv1 filters
    filters = net.layer1[0].weight.data.clone().cpu()
    grid = torchvision.utils.make_grid(filters, nrow=8, normalize=True, scale_each=True)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0)); plt.title('Conv1 Filters'); plt.axis('off'); plt.show()

    # Loss landscape interpolation
    init_w = {n: p.data.clone() for n,p in net.named_parameters()}
    final_w = {n: p.clone() for n,p in net.state_dict().items()}
    alphas = np.linspace(0,1,21)
    losses = []
    for alpha in alphas:
        for name, param in net.named_parameters():
            param.data.copy_(alpha * final_w[name] + (1-alpha) * init_w[name])
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                out = net(x)
                total_loss += criterion(out, y).item()*x.size(0)
                total += x.size(0)
        losses.append(total_loss/total)
    plt.figure()
    plt.plot(alphas, losses)
    plt.title('Loss Landscape Interpolation')
    plt.xlabel('α'); plt.ylabel('Loss'); plt.show()

    gradcam = GradCAM(net, net.layer3[1].conv2)
    img_batch, label_batch = next(iter(testloader))
    img, label = img_batch[0:1].to(device), label_batch[0:1].to(device)
    cam = gradcam(img, label[0].item())
    img_np = img.cpu().squeeze().permute(1,2,0).numpy() * 0.5 + 0.5
    plt.figure()
    plt.imshow(img_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f'Grad-CAM for class {label.item()}')
    plt.axis('off'); plt.show()

if __name__ == "__main__":
    main()
