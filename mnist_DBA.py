import numpy as np
import torch
import adversary.cw as cw
from adversary.jsma import SaliencyMapMethod
from adversary.fgsm import Attack
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
from models.mnist_model import MnistModel, MLP
from torchvision import transforms

#%reload_ext autoreload
#%autoreload 2

MNIST_UNDERCOVER_CKPT = './checkpoint/mnist_undercover.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

mlp = MLP().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


undercoverNet = MnistModel().to(device)
checkpoint = torch.load(MNIST_UNDERCOVER_CKPT, map_location=torch.device(device))
undercoverNet.load_state_dict(checkpoint['net'])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
trainiter = iter(trainloader)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)
testiter = iter(testloader)

undercover_gradient_attacker = Attack(undercoverNet, F.cross_entropy)

# construct bim adversarial samples
# --------------------train---------------------
normal_samples, adversarial_samples = [], []
for x, y in trainloader:
    x, y = x.to(device), y.to(device)
    y_pred = undercoverNet(x).argmax(dim=1)
    
    eps = 0.3
    x_adv = undercover_gradient_attacker.i_fgsm(x, y, eps=eps, alpha=1/255, iteration=int(min(eps*255 + 4, 1.25*eps*255)))
    y_pred_adv = undercoverNet(x_adv).argmax(dim=1)
    selected = (y == y_pred) & (y != y_pred_adv)
    normal_samples.append(x[selected].detach().cpu())
    adversarial_samples.append(x_adv[selected].detach().cpu())
#     break

normal_x = torch.cat(normal_samples, dim=0)
adversarial_x = torch.cat(adversarial_samples, dim=0)
normal_y = torch.zeros(normal_x.shape[0]).long()
adversarial_y = torch.ones(adversarial_x.shape[0]).long()

dba_trainloader = Data.DataLoader(Data.TensorDataset(torch.cat([normal_x, adversarial_x], dim=0),
                                           torch.cat([normal_y, adversarial_y], dim=0)), 
                                  batch_size=512, shuffle=True, num_workers=4)
dba_trainiter = iter(dba_trainloader)

# ----------------test---------------------
normal_samples, adversarial_samples = [], []
for x, y in testloader:
    x, y = x.to(device), y.to(device)
    y_pred = undercoverNet(x).argmax(dim=1)
    
    eps = 0.3
    x_adv = undercover_gradient_attacker.i_fgsm(x, y, eps=eps, alpha=1/255, iteration=int(min(eps*255 + 4, 1.25*eps*255)))
    y_pred_adv = undercoverNet(x_adv).argmax(dim=1)
    selected = (y == y_pred) & (y != y_pred_adv)
    normal_samples.append(x[selected].detach().cpu())
    adversarial_samples.append(x_adv[selected].detach().cpu())
#     break

normal_x = torch.cat(normal_samples, dim=0)
adversarial_x = torch.cat(adversarial_samples, dim=0)
normal_y = torch.zeros(normal_x.shape[0]).long()
adversarial_y = torch.ones(adversarial_x.shape[0]).long()

dba_testloader = Data.DataLoader(Data.TensorDataset(torch.cat([normal_x, adversarial_x], dim=0),
                                           torch.cat([normal_y, adversarial_y], dim=0)), 
                                  batch_size=1024, shuffle=True, num_workers=4)
dba_testiter = iter(dba_testloader)

# train the mlp
epochs = 10
for i in range(epochs):
    for x, y in dba_trainloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        _, V1 = undercoverNet(x, dba=True)
        undercover_adv = undercover_gradient_attacker.fgsm(x, y, False, 1/255)
        _, V2 = undercoverNet(undercover_adv, dba=True)
        V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
        y_pred = mlp(V)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

# test
total, correct = 0, 0
for x, y in dba_testloader:
    x, y = x.to(device), y.to(device)
    _, V1 = undercoverNet(x, dba=True)
    undercover_adv = undercover_gradient_attacker.fgsm(x, y, False, 1/255)
    _, V2 = undercoverNet(undercover_adv, dba=True)
    V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
    y_pred = mlp(V).argmax(dim=1)
    
    total += y.size(0)
    correct += y_pred.eq(y).sum().item()
print(correct / total)

