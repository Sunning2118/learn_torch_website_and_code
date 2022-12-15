import torch
import torch.nn as nn

out = torch.tensor([
    [2, 3, 4, 5], [3, 4, 5, 6], [6, 7, 8, 9], [4, 5, 7, 8]
])
target = torch.tensor(
    [1, 2, 3, 4]
)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)