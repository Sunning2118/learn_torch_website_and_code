import torch

# x1 = torch.ones(3, 3)
# print(x1)
x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
print(x.grad_fn)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
# z.backward(gradient=torch.tensor([[1,1],[1,1]]))
print(x.grad)
# print("//////////////////////////////")
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)
# print("//////////////////////////////")
# print(x.requires_grad)
# print((x ** 2).requires_grad)
# with torch.no_grad():
#     print((x ** 2).requires_grad)
# print("//////////////////////////////")
# print(x.requires_grad)
# y = x.detach()
# print(y.requires_grad)
# print(x.eq(y).all())