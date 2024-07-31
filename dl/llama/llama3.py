import sys

print(sys.argv, len(sys.argv))
if len(sys.argv) != 6:
    print("please input the model parameters, ./macs.py [hidden_dims] [decoder layers] [in_lens] [out_lens] [BW(GB/s)]\n")
    exit()

# init params
vocab_size = 128256 # llama3 extend vocab from 32005 to 128256
group_num = 4
word_dims = 4096
decoder_layers = 32
ffn_dims = 14336

word_dims = int(sys.argv[1])
decoder_layers = int(sys.argv[2])
heads = word_dims/128
ffn_dims = int((word_dims*8/3)/256 + 0.5) * 256
in_len = int(sys.argv[3])
out_len = int(sys.argv[4])
bw = float(sys.argv[5])

def MACs(len):
    MAC_Q = word_dims*word_dims # after 1st time, generate 1token update.
    MAC_KV = 2*word_dims*word_dims/group_num
    MAC_QKT = 128*len*heads # should add to previous token.
    MAC_Z = len*128*heads
    MAC_LINEAR = word_dims*word_dims
    MAC_FFN = word_dims*ffn_dims*2+ffn_dims*word_dims
    MAC_DEC = decoder_layers*(MAC_Q + MAC_KV + MAC_QKT + MAC_Z + MAC_LINEAR + MAC_FFN)
    MAC_OUT_LINEAR = vocab_size*word_dims
    return  MAC_DEC + MAC_OUT_LINEAR

def total_mac(in_len, out_len):
    sum = 0
    for i in range(1, out_len+1):
        sum += MACs(in_len+i)
        i += 1  
    return sum

mac = total_mac(in_len, out_len)

def weight(len, wt_type='INT8', kvcache_type='FP16'):
    if wt_type == 'FP32':
        dlen = 4
    elif wt_type in ['FP16','BF16']:
        dlen = 2
    elif wt_type == 'INT8':
        dlen = 1
    elif wt_type == 'INT4':
        dlen = 0.5
    if kvcache_type == 'FP16':
        kvtype = 2
    elif kvcache_type == 'INT8':
        kvtype = 1
    W_Q = word_dims*word_dims
    W_KV = word_dims*word_dims/group_num
    W_QKV = W_Q + W_KV*2
    W_kvcache = (len*word_dims/group_num + word_dims/group_num)*2*kvtype # treat k,v as wt, read all and write once (4K/8)BW
    W_LINEAR = word_dims*word_dims
    W_FFN = (word_dims*ffn_dims)*2 + word_dims*ffn_dims # up, gate, down
    W_NORM = word_dims*2 # input norm and post attentation norm
    W_DECODER = decoder_layers * ( dlen*(W_QKV + W_FFN + W_LINEAR + W_NORM)+ W_kvcache)
    #W_DECODER = decoder_layers * ( dlen*(W_QKV + W_FFN )+ W_kvcache)
    # print(W_DECODER/decoder_layers)
    W_POS_EMBD = word_dims*dlen
    W_OUT_LINEAR = vocab_size*word_dims*dlen
    return W_POS_EMBD + W_DECODER + W_OUT_LINEAR

def total_weight(in_len, out_len, wt_type='INT8', kv_type='FP16'):
    sum = 0
    for i in range(in_len, in_len+out_len):
        sum += weight(i,wt_type, kv_type)
        i += 1
    return sum

wt_size = total_weight(in_len, out_len)
wt_size_int4 = total_weight(in_len, out_len, 'INT4', 'INT8')

#print("input %d words generate %d words will cost %.3fTMACs"%(in_len, out_len, mac/2**40))
#print("under %.1fTB/s BW, perf is %.1ftoken/s@FP8 and %.1ftokens@FP16"%(bw, out_len*bw/(mac/2**40), out_len*bw*0.5/(mac/2**40)))
#print("FP32 model weight size is %.2fGB, FP16 is %.2fGB, INT8 is %.2fGB"%(weight(1,'FP32')/2**30,weight(1,'FP16')/2**30, weight(1,'INT8')/2**30))
print("llama parameters are dims=%d, layers=%d, heads=%d, ffn_nodes=%d"%(word_dims, decoder_layers, heads, ffn_dims))
print("input %d words generate %d words will cost %.3fTMACs"%(in_len, out_len, mac/2**40))
print("under %.2fGB/s BW, ideal perf is %.2ftoken/s@FP8orINT8, %.2ftokens/s@INT4"%(bw, out_len*bw/(wt_size/2**30), out_len*bw/(wt_size_int4/2**30)))
