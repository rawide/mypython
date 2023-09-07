import math
import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(-128, 127, 1)
x = np.arange(-448., 448., 1.)
exp = math.e**x
ln2 = 0.6931471805599453
new_x = x/ln2
# print(new_x.size)

def lookup_softmax(lut, index):
    return lut.get(np.float16(index), 1)

softmax_index = np.arange(int(new_x[0])-1., int(new_x[-1])+1., 1.)
# print(LUT_index)
softmax_index += 0.5
# print(LUT_index)
softmax_lut = {i: (2.**i) for i in softmax_index}
# print(softmax_lut)


def lienar_intepolate(x0, y0, x1, y1, x):
    return y0 + (x-x0)*(y1-y0)/(x1-x0)

def calc_appr_exp(i):
    if i < 0:
        if int(i) - i <= 0.5:
            x0 = int(i)-0.5
            y0 = lookup_softmax(softmax_lut, x0)
            x1 = int(i)
            y1 = 2.**x1
        else:
            x0 = int(i)-1.
            y0 = 2.**x0
            x1 = int(i)-0.5
            y1 = lookup_softmax(softmax_lut, x1)
    else:
        if i - int(i) <= 0.5:
            x0 = int(i)
            y0 = 2.**x0
            x1 = int(i)+0.5
            y1 = lookup_softmax(softmax_lut, x1)
        else:
            x0 = int(i)+0.5
            y0 = lookup_softmax(softmax_lut, x0)
            x1 = int(i+1)
            y1 = 2.**x1
    return lienar_intepolate(x0, y0, x1, y1, i)


dec_softmax_lut = {np.float16(i): np.float16(2.**i) for i in np.arange(-0.99, 1, 0.01)}
# print(dec_softmax_lut)

def calc_appr_exp2(i):
    dec = i - int(i)
    dec = round(dec, 2)
    e = int(i)
    if dec == 1.0:
        e += 1
    elif dec == -1.0:
        e -= 1
    return 2.**e * lookup_softmax(dec_softmax_lut, dec)


apprExp1 = [calc_appr_exp(i) for i in new_x]
err1 = [abs(100*(exp[i]-apprExp1[i])/exp[i]) for i in range(0, exp.size, 1)]

apprExp2 = [calc_appr_exp2(i) for i in new_x]
err2 = [abs(100*(exp[i]-apprExp2[i])/exp[i]) for i in range(0, exp.size, 1)]
# print(apprExp2)
# print(exp)
# print(apprExp2[3], exp[3], err2[3], x[3], new_x[3])
# print(err2)
# plt.plot(x, err1)
plt.plot(x, err2)
plt.grid(True)
plt.show()
