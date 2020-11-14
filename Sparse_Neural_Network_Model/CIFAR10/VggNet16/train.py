import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import os
import argparse
import numpy
from utils import progress_bar
from model import VggNet16
import pdb


parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=3000, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
gpu = "0"           # which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss = 1000
best_acc = 0        # best test accuracy
start_epoch = 0     # start from epoch 0 or last checkpoint epoch
flod = args.flod
work = True
weight_decay = 0.000


# Data
def data_prepare():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    trainset = torchvision.datasets.CIFAR10(root='../CIFAR10', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='../CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


# Model
def model_prepare(work):
    print('==> Building model..')
    global best_acc
    global start_epoch

    if work == True:
        net = VggNet16()
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    # TO Check the check point.
    if args.resume:                           #False
        print('==> Resuming model from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_flod' + str(flod) + 'mode' + mode + '.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # torch.optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.1, last_epoch=-1)
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100, 200, 200, 300], gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    # checkpoint = torch.load('./finial/flod1.t7')
    # net.load_state_dict(checkpoint['net'])
    # return net, optimizer, criterion
    return net, optimizer, scheduler, criterion


def train(epoch, dataloader, net, optimizer, criterion, vali=True):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    global tr_loss
    net.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # if batch_id < (12800 / args.batch_size):
        num_id += 1
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                     % (train_loss / (batch_id + 1), 100. * correct / total, correct, total))
        # else:
        #     print('End of the train')
        #     break
    if vali is True:
        tr_loss = train_loss / num_id
    return train_loss / num_id, 100. * correct / total


def test(epoch, dataloader, net, criterion, vali=True):
    """Validation and the test."""
    global best_acc
    net.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            # if batch_id < (2560 / args.batch_size):
            num_id += 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())

            test_loss += loss.item()
            _, predicted = outputs.max(1)  # judge max elements in predicted`s Row(1:Row     0:Column)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # judge how many elements same in predicted and targets
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_id + 1), 100. * correct / total, correct, total))
            # else:
            #     print('End of the test')
            # break
    if vali is True:
        # Save checkpoint.
        acc = 100. * correct / total
        if acc >= best_acc:
            print('Saving:')
            state1 = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('best_acc'):
                os.mkdir('best_acc')
            torch.save(state1, './best_acc/flod''.t7')
            best_acc = acc
            best_acc1 = open('VggNet16.txt', 'w')
            best_acc1.write(str(best_acc))
            best_acc1.close()
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':

    trainloader, testloader = data_prepare()
    # net, optimizer, criterion = model_prepare(work)
    net, optimizer, scheduler, criterion = model_prepare(work)
    train_list0, train_list1 = [], []
    test_list0, test_list1 = [], []
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        train_loss, train_acc = train(epoch, trainloader, net, optimizer, criterion)
        test_loss, test_acc = test(epoch, testloader, net, criterion)

        scheduler.step(tr_loss)
        lr = optimizer.param_groups[0]['lr']

        train_list0.append(train_loss)
        train_list1.append(train_acc)
        test_list0.append(test_loss)
        test_list1.append(test_acc)
        if lr >= 5e-7:
            pass
        elif lr < 5e-7:
            print('Saving:')
            state3 = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('finial'):
                os.mkdir('finial')
            torch.save(state3, './finial/flod''.t7')
            train_array0 = numpy.array(train_list0)
            train_array1 = numpy.array(train_list1)
            test_array0 = numpy.array(test_list0)
            test_array1 = numpy.array(test_list1)
            plt.figure(1)
            plt.subplot(2, 2, 1)
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.plot([i for i in range(epoch+1)], train_array0, '-')
            plt.subplot(2, 2, 2)
            plt.xlabel('epoch')
            plt.ylabel('train acc')
            plt.plot([i for i in range(epoch+1)], train_array1, '-')
            plt.subplot(2, 2, 3)
            plt.xlabel('epoch')
            plt.ylabel('test loss')
            plt.plot([i for i in range(epoch+1)], test_array0, '-')
            plt.subplot(2, 2, 4)
            plt.xlabel('epoch')
            plt.ylabel('test acc')
            plt.plot([i for i in range(epoch+1)], test_array1, '-')
            plt.savefig("VggNet16.jpg")
            plt.show()
            print('OVER')
            break