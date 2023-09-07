import torch
from torch.nn import functional as F
import math
import numpy as np

ln2 = 0.6931471805599453
high, low = 127, -128
# x = low + (high-low)*torch.rand(5)
x = torch.tensor([0.1, 0.2, 2, 0.15, 0.16])
sfm_x = F.softmax(x, dim = -1)
print(f"original tensor is\n{x}\nthe softmax is\n{sfm_x}")

x_max, _ = torch.max(x, dim=0)
# print(x_max)
x_avg = x - x_max
z = torch.tensor([int(-1.*i/ln2) for i in x_avg]) 
p = x_avg + z*ln2
print(z,p)

LUT = {np.float16(i): np.float16(math.e**i) for i in np.arange(-0.693, 0.001, 0.001)}
# print(LUT)

def calc_appr_exp(i):
    dec = round(i.item(), 3)
    return LUT.get(np.float16(dec))

exp_p = torch.tensor([calc_appr_exp(i) for i in p])
Lp = torch.tensor([0.3585*(i+1.353)**2+0.344 for i in p])
# # exp_p_sz = torch.empty(p.size(0))
# # for (i, j) in (exp_p, z):
# #     exp_p_sz[i] = shift_right(i, j)
# exp_p_sz = torch.tensor([float_shift_right(i, j) for (i, j) in zip(exp_p, z)])
print(exp_p, Lp)

'''
x_norm = F.normalize(x, p=math.e, dim = 0)
print(f"after norm, tensor is\n{x_norm}\nsoftmax(x_norm)=\n{F.softmax(x_norm, dim = -1)}\n{x_norm/x}")
LUT = {np.float16(i): np.float16(math.e**i) for i in np.arange(-0.999, 1., 0.001)}
# print(LUT)

def calc_appr_exp(i):
    dec = round(i, 3)
    return LUT.get(np.float16(dec))

appr_exp = [calc_appr_exp(x_norm[i].item()) for i in range(0, x_norm.size(0))]
sum = 0
for i in range (0, len(appr_exp)):
    sum += appr_exp[i]

print(sum, appr_exp)

sfm_norm_x = appr_exp/sum
print(sfm_norm_x)
'''