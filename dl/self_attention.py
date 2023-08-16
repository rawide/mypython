import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512  # 模型的维度
num_heads = 8  # 注意力头的数量
d_k = d_model // num_heads  # 每个头的维度

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
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割为多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力权重
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
        
        # 最后的线性变换
        output = self.W_o(output)
        
        return output

def subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask.unsqueeze(0)

sequence_length = 3
batch_size = 1

# 示例输入
input_tensor = torch.rand(batch_size, sequence_length, d_model)

# 创建掩码
mask = subsequent_mask(sequence_length)
# print(mask)

# 使用多头自注意力
mha = MultiHeadAttention(d_model, num_heads)
output = mha(input_tensor, input_tensor, input_tensor, mask)


print(input_tensor)
print(output)