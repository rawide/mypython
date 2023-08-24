import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 8
num_heads = 2
seq_len = 3
num_block = 4
d_k = d_model // num_heads

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, input, mask=None):
        batch_size = input.size(0)
        
        # 线性变换
        Q = self.W_q(input)
        K = self.W_k(input)
        V = self.W_v(input)
        
        # 分割为多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力权重
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 1, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        
        # 最后的线性变换
        output = self.W_o(output)
        
        return output
    
class decoder:
    mha = []
    block_output = []
    def __init__(self, input, blocks, mask):
        self.input = input
        self.blocks = blocks
        self.output = input
        self.mask = mask
        for i in range(0, blocks):
            decoder.mha.append(MultiHeadAttention(d_model, num_heads))

    def mha_output(self):
        for i in range (0, self.blocks):
            self.output = decoder.mha[i](self.output, self.mask)
            decoder.block_output.append(self.output)
        return self.output
    

def subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0)

sequence_length = 4
input_tensor_t0 = torch.rand(1, sequence_length, d_model)
mask_t0 = subsequent_mask(sequence_length)
llama = decoder(input_tensor_t0, num_block, mask_t0)
output_t0 = llama.mha_output()
print(output_t0)
print(llama.block_output)