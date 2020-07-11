'''
@Author: your name
@Date: 2020-07-08 20:06:49
@LastEditTime: 2020-07-10 21:01:54
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \LeNet5\LeNet_5.py
'''
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import datetime

start = datetime.datetime.now()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 5  #迭代次数
num_classes = 10  #最后一层的输出，即分类类别，MNIST共10类
batch_size = 100  #每个批次加载的样本数量
learning_rate = 0.001  #学习率

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
"""
train=True下载训练集数据，False下载测试集数据,
download（bool，可选）–如果为true，则从internet下载数据集并将其放在根目录中。
如果数据集已下载，则不会再次下载，下载数据时建议科学上网
transform（可调用，可选）–接受PIL图像并返回已转换版本的函数/转换。
E、 g，变换。随机裁剪
"""
test_dataset = torchvision.datasets.MNIST(root='data/',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader
"""
torch.utils.data.DataLoader：将自定义的dataset根据batch_size大小、
是否shuffle等封装成一个Batch_Size大小的Tensor，用于后面的训练
"""
train_loader = Data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
#shuffle=True打乱数据
test_loader = Data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

"""
创建LeNet-5网络结构，使用nn.Module方法便于管理参数，继承nn.Module转换成类的形式
"""
class LetNet5(nn.Module):
    def __init__(self, num_clases=10):
        super(LetNet5, self).__init__()
        """定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，
        外层补上了两圈0，因为输入的是32*32，批量归一化(BN算法)，激活函数，最大池化
        BN算法其作用可以加快模型训练时的收敛速度，使得模型训练过程更加稳定，
        避免梯度爆炸或者梯度消失。并且起到一定的正则化作用，几乎代替了Dropout。
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        """第二个卷积层，6个输入，16个输出，5*5的卷积filter，
        批量归一化(里面的数与输出相同)，激活函数，最大池化"""
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        """第一个全连接层"""
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU()  #激活函数ReLU
        )
        """第二个全连接层"""
        self.full_con1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()  #激活函数ReLU
        )
        """最后一个全连接层，原本是高斯连接，这里改成全连接层"""
        self.full_con2 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.LogSoftmax()  #激活函数LogSoftmax
        )
    """按顺序搭建LeNet-5网络，前馈网络"""
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)  #
        out = self.full_con1(out)
        out = self.full_con2(out)
        return out

"""将CNN网络赋给model，在下面的模型训练中调用"""
model = LetNet5(num_classes).to(device)
"""损失函数为交叉熵"""
criterion = nn.CrossEntropyLoss()
"""选择Adam优化器，学习率设为0.001"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
"""训练模型"""
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  #图像数据保存（CPU中或GPU中，看devise）
        labels = labels.to(device)  #标签数据保存（CPU中或GPU中，看devise）

        # Forward pass
        outputs = model(images)  #CNN训练模型
        loss = criterion(outputs, labels)  #损失函数交叉熵

        # Backward and optimize
        optimizer.zero_grad()  #梯度清零
        loss.backward()  #BP过程，反馈回路
        optimizer.step()  #更新梯度

        if (i + 1) % 100 == 0:
            """输出的格式，{}里的内容依次对应format里的内容"""
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

"""
测试模型，评估模式（batchnorm使用移动均值/方差，而不是mini-batch均值/方差）
"""
"""
在model(test_datasets)之前，需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值。
这是model中含有batch normalization(BN)层所带来的的性质。
model.train() ：启用 BatchNormalization 和 Dropout
model.eval() ：不启用 BatchNormalization 和 Dropout
"""
model.eval()
with torch.no_grad():
    correct = 0  #正确预测数
    total = 0  #总数，即标签，数字0~9
    for images, labels in test_loader:  #从测试集中读取图像和标签
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  #用模型进行预测输出
        _, predicted = torch.max(outputs.data, 1)  #选择CNN输出结果最大的为预测值
        total += labels.size(0)  #依次令标签+1
        correct += (predicted == labels).sum().item()  #若预测值等于标签数，则正确数+1

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

"""
保存模型
torch.save(model.state_dict(), PATH)
加载模型
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
一定要使用model.eval()来固定dropout和归一化层，否则每次推理会生成不同的结果
如果使用nn.DataParallel在一台电脑上使用了多个GPU，那么加载模型的时候也必须先进行nn.DataParallel
注意，load_state_dict()需要传入字典对象，因此需要先反序列化state_dict再传入load_state_dict()
参考链接：https://blog.csdn.net/LXYTSOS/article/details/90639524
https://www.cnblogs.com/qinduanyinghua/p/9311410.html
"""
torch.save(model.state_dict(), 'LetNet-5.ckpt')  #保存模型

end = datetime.datetime.now()
print("Running time: %s seconds" % (end - start))