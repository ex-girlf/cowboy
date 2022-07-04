import torch.nn
import torchvision
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch.nn import CrossEntropyLoss

# 准备数据集
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

# length长度

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练数据集的长度为:{}".format(train_dataset_size))
print("验证数据集的长度为:{}".format(test_dataset_size))

# 加载数据
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 创建网络模型
model = Wmodel()
write = SummaryWriter("../log_train")
# 损失函数
loss_fn = CrossEntropyLoss()

# 优化函数
# 1e-2 = 1 * 10^-2
optimzer = SGD(model.parameters(), lr=1e-2)

# 训练次数
totol_train_count = 0
# 验证次数
totol_test_count = 0

epoch = 20
# 反向传播
for i in range(epoch):
    print("----第{}循环迭代开始---".format(i + 1))
    for data in train_loader:
        imgs, targets = data
        out = model(imgs)
        loss = loss_fn(out, targets)

        #优化器优化模型
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        totol_train_count += 1
        if totol_train_count % 100 == 0:
            print("训练次数：{},Loss为{}".format(totol_train_count, loss.item()))
            write.add_scalar("train_loss", loss.item(), totol_train_count)

    #测试部分
    total_test_loss = 0
    total_acc_rate = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            acc = (output.argmax(1) == targets).sum()
            total_acc_rate += acc
    print("验证集的loss为{}".format(total_test_loss))
    print("验证集的正确率为{}".format(total_acc_rate/test_dataset_size))
    write.add_scalar("test_loss", total_test_loss, totol_test_count)
    write.add_scalar("test_accuary", total_acc_rate/test_dataset_size, totol_test_count)
    totol_test_count += 1

    torch.save(model, "mode_{}.pth".format(i))
    print("模型已保存")

write.close()