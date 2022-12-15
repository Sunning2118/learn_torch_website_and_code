import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

net = nn.Linear(3, 4)  # 一层的网络，也可以算是一个计算图就构建好了
input = Variable(torch.randn(1, 2, 3), requires_grad=True)  # 定义一个图的输入变量
output = net(input)  # 最后的输出
loss = torch.sum(output)  # 这边加了一个sum() ,因为被backward只能是标量
loss.backward()  # 到这计算图已经结束，计算图被释放了
print(output)
print(loss)
