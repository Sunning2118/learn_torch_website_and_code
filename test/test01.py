# test = []
# test.extend(range(10))
# print(test)
# print(type(range(10)))
# for i in range(3, 10):
#     print(i)
import torch

a = torch.rand(3, 3, dtype=torch.float)
b = torch.rand(3, 3, dtype=torch.double)
c = torch.rand(3, 3, dtype=torch.long)
