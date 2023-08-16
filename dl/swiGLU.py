import torch
import matplotlib.pyplot as plt
import numpy as np
import math

# 定义输入数据
x = torch.linspace(-10, 10, 400)
y = torch.linspace(-10, 10, 400)

# Swish函数
def swish(input):
    return input * torch.sigmoid(input)
y_swish = swish(x)

def swiGLU(x, y, beta=1.0):
    return x * torch.sigmoid(beta*y)

# 绘制
plt.figure(figsize=(12, 8))
j=0
win = np.arange(-1, 1.1, 0.2)
wincont = int(math.sqrt(win.size))
for i in win:
    y_swiGLU = swiGLU(x, y, i)
    j+=1
    plt.subplot(wincont, wincont+1, j)
    plt.plot(x.numpy(), y_swiGLU.numpy(), label='swiGLU %.2f'%i)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
