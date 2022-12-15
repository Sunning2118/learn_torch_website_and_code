from __future__ import print_function

import torch

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# # 对其中一个进行加法操作, 另一个也随之被改变:
# a.add_(1)
# print(a)
# print(b)
# # 将Numpy array转换为Torch Tensor:
# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)
x = torch.tensor([
    [1., 2., 3.],
    [2., 3., 4.]
])


if torch.cuda.is_available():
    # 定义一个设备对象, 这里指定成CUDA, 即使用GPU
    device = torch.device("cuda")
    # 直接在GPU上创建一个Tensor
    y = torch.ones_like(x, device=device)
    # 将在CPU上面的x张量移动到GPU上面
    print(y.type())
    x = x.to(device)
    # x和y都在GPU上面, 才能支持加法运算
    z = x + y
    # 此处的张量z在GPU上面
    print(z)
    print(z.type())
    # 也可以将z转移到CPU上面, 并同时指定张量元素的数据类型
    print(z.to("cpu", torch.double))
