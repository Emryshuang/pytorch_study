import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y1 = x + 2
print('\ny1:\n',y1)
print(y1.grad_fn)

z1 = y1 * y1 * 3

out1 = z1.mean()
print(z1, out1)

out1.backward()
print(x.grad)

y2 = 3 * x

print('\ny2:\n',y2)
print(y2.grad_fn)

z2 = y2 * y2 *3
out2 = z2.mean()
print(z2, out2)

out2.backward()
print(x.grad)