from __future__ import print_function
import torch

x = torch.empty(2, 2)
y = torch.ones(5, 3)
z = torch.zeros(5, 3, dtype=torch.float)
print(x)
print(x.type())
print(y)
print(x.type())
print(z)
print(x.type())
k = torch.tensor([2.1, 3.1, 4.1])
print(k.size())

x = x.new_ones(5, 3, dtype=torch.float)
print(x)

l = torch.randn_like(x, dtype=torch.float)
print(l)

print(x.type())
print(y.type())
# 第一种加法方式:
print(x+y)
# 第二种加法方式:
print(torch.add(x, y))
# 第三种加法方式:
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 第四种加法方式: in-place (原地置换)
y.add_(x)
print(y)
# 改变张量的形状: torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
# 如果张量中只有一个元素, 可以用.item()将值取出, 作为一个python number
x = torch.randn(1)
print(x)
print(x.item())