import torch
from torch.nn import functional as F

d_model = 32
num_head = 8
seq_len = 3
num_block = 1
d_k = d_model // num_head
batch_size = 1

src = torch.rand(1, seq_len, d_model)
input_t1_delta = torch.rand(1, 1, d_model)
input_t1 = torch.concat((src, input_t1_delta), dim = 1)

class DecBlock:
    def __init__(self, d_model, num_heads, wq, wk, wv, wo):
        self.Wq = wq
        self.Wk = wk
        self.Wv = wv
        self.Wo = wo
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
    
    def calcQKV(self, input):
        self.Q = torch.matmul(input, self.Wq)
        self.K = torch.matmul(input, self.Wk)
        self.V = torch.matmul(input, self.Wv)

    def updateQKV(self, input_delta):
        Q_delta = torch.matmul(input_delta, self.Wq)
        K_delta = torch.matmul(input_delta, self.Wk)
        V_delta = torch.matmul(input_delta, self.Wv)
        self.Q = torch.concat((self.Q, Q_delta), dim = -2)
        self.K = torch.concat((self.K, K_delta), dim = -2)
        self.V = torch.concat((self.V, V_delta), dim = -2)

    def mhaOutput(self):
        Q = self.Q.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.K.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.V.view(1, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2,-1))/self.d_k**0.5
        mask = torch.triu(torch.ones(Q.shape[-2], Q.shape[-2]), diagonal=1).bool()
        mask = mask.unsqueeze(0)
        attn = attn.masked_fill(mask == 1, float('-inf'))
        attn = F.softmax(attn, dim = -1)
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(1, -1, self.d_model)
        return output

wq, wk, wv, wo = [torch.rand(num_block, batch_size, d_model, d_model) for _ in range(0,4)]

llama = [DecBlock(d_model, num_head, wq[i], wk[i], wv[i], wo[i]) for i in range(0, num_block)]
new_llama = [DecBlock(d_model, num_head, wq[i], wk[i], wv[i], wo[i]) for i in range(0, num_block)]

llama_output = input_t1
for i in range(0, num_block):
    llama[i].calcQKV(llama_output)
    llama_output = llama[i].mhaOutput()
print(f"{input_t1}\n{llama_output}")

# use K V cache opt.
new_llama_output = src
for i in range(0, num_block):
    new_llama[i].calcQKV(new_llama_output)
    new_llama_output = new_llama[i].mhaOutput()

new_llama_output = input_t1_delta
for i in range(0, num_block):
    new_llama[i].updateQKV(new_llama_output)
    new_llama_output = new_llama[i].mhaOutput()[0][-1]
    new_llama_output = new_llama_output.unsqueeze(0)
print(f"{input_t1_delta}\n{new_llama_output}")




'''
# w_q=[], w_k=[], w_v = []
# for i in range(0, num_block):
#     w_q[i], w_k[i], w_v[i] = [torch.rand(1, d_model, d_model) for _ in range(0,3)]
w_q, w_k, w_v = [torch.rand(num_block, 1, d_model, d_model) for _ in range(0, 3)]

# def calc_QKV(input, wq, wk, wv, i):
#     Q[i] = torch.matmul(input, wq)
#     K[i] = torch.matmul(input, wk)
#     V[i] = torch.matmul(input, wv)

def calc_QKV(input, i):
    Q = torch.matmul(input, w_q[i])
    K = torch.matmul(input, w_k[i])
    V = torch.matmul(input, w_v[i])
    return Q,K,V

def calc_head(Q, K, V, mask):
    Q = Q.view(1, -1, num_head, d_k).transpose(1, 2)
    K = K.view(1, -1, num_head, d_k).transpose(1, 2)
    V = V.view(1, -1, num_head, d_k).transpose(1, 2)
    attn = torch.matmul(Q, K.transpose(-2,-1))/d_k**0.5
    attn = attn.masked_fill(mask == 1, float('-inf'))
    attn = F.softmax(attn, dim = -1)
    output = torch.matmul(attn, V)
    output = output.transpose(1, 2).contiguous().view(1, -1, d_model)
    return output

def calc_decoder(input, len):
    input = input
    output = torch.empty(num_block, 1, len, d_model)
    for i in range(0, num_block):
        # calc_QKV(input, w_q[i], w_k[i], w_v[i], i)
        Q,K,V = calc_QKV(input, i)
        mask = torch.triu(torch.ones(len, len), diagonal=1).bool()
        mask = mask.unsqueeze(0)
        input = calc_head(Q, K, V, mask)
        output[i] = input
    return output

Q,K,V = [torch.empty(num_block, 1, seq_len, d_model) for _ in range(0, 3)]
def store_QKV(input, len):
    global Q,K,V
    input = input
    output = torch.empty(num_block, 1, len, d_model)
    for i in range(0, num_block):
        Q[i],K[i],V[i] = calc_QKV(input, i)
        mask = torch.triu(torch.ones(len, len), diagonal=1).bool()
        mask = mask.unsqueeze(0)
        input = calc_head(Q[i],K[i],V[i],mask)

store_QKV(src, seq_len)


output_0 = calc_decoder(input_t1, seq_len+1)
print(f"{input_t1_delta}\n{input_t1}")
print(output_0[num_block-1])

delta = torch.empty(num_block, 1, 1, d_model)
Q_c, K_c, V_c = [torch.concat((x, delta), dim = 2) for x in [Q, K, V]]


for i in range(0, num_block):
    Q_c[i] = torch.matmul(input_t1_delta, w_q[i])
    K_c[i] = torch.matmul(input_t1_delta, w_k[i])
    V_c[i] = torch.matmul(input_t1_delta, w_v[i])


# Q1, K1, V1 = [torch.concat]
# input = src
# output = torch.empty(num_block, 1, seq_len+1, d_model)
# for i in range(0, num_block):
#     calc_QKV(input, w_q[i], w_k[i], w_v[i], i)
#     input = calc_head(Q[i], K[i], V[i])
#     output[i] = input

# print(src)
# print(output[num_block-1])


# w_q, w_k, w_v = [torch.rand(1, d_model, d_model) for _ in range(0,3)]

# calc full martix.
Q = torch.matmul(input_t1, w_q)
K = torch.matmul(input_t1, w_k)
V = torch.matmul(input_t1, w_v)
atten = torch.matmul(Q, K.transpose(-2, -1))/d_k**0.5
atten = F.softmax(atten, dim = -1)
output = torch.matmul(atten, V)

# use cache line opt
Q_opt = torch.matmul(input, w_q)
K_opt = torch.matmul(input, w_k)
V_opt = torch.matmul(input, w_v)
Q_delta = torch.matmul(input_t1_delta, w_q)
K_delta = torch.matmul(input_t1_delta, w_k)
V_delta = torch.matmul(input_t1_delta, w_v)

K_new = torch.concat((K_opt, K_delta), dim = 1) # K should be totally update.
Q_new = torch.concat((Q_opt, Q_delta), dim = 1)
V_new = torch.concat((V_opt, V_delta), dim = 1)
atten_new = torch.matmul(Q_new, K_new.transpose(-2, -1))/d_k**0.5
atten_new = F.softmax(atten_new, dim = -1)
output_new = torch.matmul(atten_new, V_new)

print(f"{input_t1}\n{input_t1_delta}\n{output}\n{output_new}")
'''