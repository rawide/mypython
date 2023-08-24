import torch
from torch.nn import functional as F
import math

Dim_embed = 5
Dim_ffn = 5
dec_layers = 3
Max_len = 10
dK = math.sqrt(Dim_embed)

def head_output(input, wq, wk, wv, linear, mask_en=False):
    Q = torch.mm(input, wq)
    K = torch.mm(input, wk)
    V = torch.mm(input, wv)
    QKt = torch.mm(Q, K.t())
    if mask_en == True :
        mask = torch.tril(torch.ones(input.size()[0], input.size()[0]), diagonal=1)
        mask = mask.eq(0)
        QKt.masked_fill_(mask, float('-inf'))
    P = F.softmax(QKt/dK, dim = 0)
    Z = torch.mm(P, V)
    return torch.mm(Z, linear) + input

def multi_head_output(input, heads, wq, wk, wv, linear):
    for i in range(0, heads):
        Q = torch.mm(input, wq[i])
        K = torch.mm(input, wk[i])
        V = torch.mm(input, wv[i])
        QKt = torch.mm(Q, K.t())
        P = F.softmax(QKt, dim = 0)
        Zi = torch.mm(P, Z)
        Z = torch.concat((Z, Zi), dim=1)
    return torch.mm(Z, linear)

def generate_input(new_lines, src_seq = torch.rand(1, Dim_embed)):
    x = src_seq
    for i in range(0, new_lines):
        row = torch.rand(1, Dim_embed)
        x = torch.concat((x, row), dim=0)
    return x

def generate_output(src_seq, WQ, WK, WV, LINEAR, mask_en=False):
    x = head_output(src_seq, WQ[0], WK[0], WV[0], LINEAR[0], mask_en)
    for i in range(1, WQ.size()[0]):
        x = head_output(x, WQ[i], WK[i], WV[i], LINEAR[i], mask_en)
    return x

# generate Wq, Wk and wv
Wq, Wk, Wv, head_linear = [torch.rand(dec_layers, Dim_embed, Dim_embed) for _ in range(0,4)]
# print(f"{Wq[0]}\n{Wq[1]}\n{Wk[0]}\n{Wk[1]}\n{Wv[0]}\n{Wv[1]}")

# generate input and output seq
src = generate_input(3)
token1 = generate_input(1, src)
token2 = generate_input(1, token1)
# print(f"{token1}\n{token2}")

# calculate the decoder output.
out1 = generate_output(token1, Wq, Wk, Wv, head_linear)
out2 = generate_output(token2, Wq, Wk, Wv, head_linear)
print(f"{out1}\n{out2}")

supply_token1 = torch.zeros((Max_len-token1.size()[0]), Dim_embed)
supply_token2 = torch.zeros((Max_len-token2.size()[0]), Dim_embed)
max_token1 = torch.concat((token1, supply_token1),dim=0)
max_token2 = torch.concat((token2, supply_token2),dim=0)
# print(f"{max_token1}\n{max_token2}")
max_out1 = generate_output(max_token1, Wq, Wk, Wv, head_linear, True)
max_out2 = generate_output(max_token2, Wq, Wk, Wv, head_linear, True)
print(f"{max_out1}\n{max_out2}")