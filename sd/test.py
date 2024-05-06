import sys
import torch
import inspect

# sample = torch.tensor([[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]])
# down_block_res_samples = (sample,)
# out_state = ()
# sample = 2*sample
# out_state = out_state + (sample,)
# sample = 3*sample
# out_state = out_state + (sample, )
# down_block_res_samples += out_state

# print(out_state)
# print(down_block_res_samples)

# temb = torch.randn(1,4,8,8)
# # temb = temb[:,:,None,None]
# # print(temb.shape[1])
# # sample = torch.ones(1,32,4,4)
# # print(temb+sample)
# t1 = (1,2,3)
# t2 = (4,5)
# t3 = (6,7,8,9)
# t = t1+t2+t3
# print(t)
SS = 20
S = 0
def gm_overflow_check(size):
    global S
    if size > SS - S:
        return True
    else:
        print("current gm used = %.2fMB, required = %.2fMB, remiand = %.2fMB"%(S, size, (SS-S-size)))
        S += size
        return False

def gm_remove(size):
    global S
    stack = inspect.stack()
    calling = stack[2].function
    print(calling)
    if S < size:
        print("[ERR] release GM size exceed remaind! current used %.2fMB, released %.2fMB"%(S, size))
    else:
        S -= size
        print("gm size removed %.2fMB, still remained %.2fMB"%(size, S))

class A():
    def __init__(
            self
    ):
        gm_overflow_check(1)

    def release(
            self
    ):
        gm_remove(1)

def func(input):
    input.release()
    a = A()
    return a

aa = A()
b = func(aa)

print(S)