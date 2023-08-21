import torch
import matplotlib.pyplot as plt

# 定义输入数据
x = torch.linspace(-10, 10, 400)
y = torch.linspace(-10, 10, 400)

# ReLU函数
# relu = torch.nn.ReLU()
# y_relu = relu(x)

# Swish函数
def swish(input):
    return input * torch.sigmoid(input)
y_swish = swish(x)

def swiGLU(x, y, beta=1.0):
    return x * torch.sigmoid(beta*y)
y_swiGLU = swiGLU(x, y, 0.67)

# GLU函数
# 注意：GLU需要2倍的输入大小，因为它将输入分为两部分
x_glu = torch.cat((x, x), dim=0)
glu = torch.nn.GLU()
y_glu = glu(x_glu)

# 绘制
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x.numpy(), y_swish.numpy(), label='Swish')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x_glu.numpy()[:400], y_glu.numpy(), label='GLU')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x.numpy(), y_swiGLU.numpy(), label='swiGLU')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.legend()
plt.grid(True)

# plt.subplot(2, 3, 1)
# plt.plot(x.numpy(), y_relu.numpy(), label='ReLU')
# plt.legend()
# plt.grid(True)

plt.tight_layout()
plt.show()
