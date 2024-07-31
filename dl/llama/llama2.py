import sys

vocab_size = 46336

print(sys.argv, len(sys.argv))
if len(sys.argv) != 8:
    print("please input the model parameters, ./macs.py [words size 46336] [hidden_dims] [decoder layers] [in_lens] [out_lens] [BW(GB/s)] [preccision(INT8/FP16)]\n")
    exit()

vocab_size = int(sys.argv[1])
word_dims = int(sys.argv[2])
decoder_layers = int(sys.argv[3])
heads = word_dims/128
ffn_dims = int((word_dims*8/3)/256 + 0.5) * 256
in_len = int(sys.argv[4])
out_len = int(sys.argv[5])
bw = float(sys.argv[6])
precision = sys.argv[7]

def MACs(len):
    MAC_QKV = word_dims*word_dims*3 # after 1st time, generate 1token update.
    MAC_QKT = 128*len*heads # should add to previous token.
    MAC_Z = len*128*heads
    MAC_LINEAR = word_dims*word_dims
    MAC_FFN = word_dims*ffn_dims*2+ffn_dims*word_dims
    MAC_DEC = decoder_layers*(MAC_QKV + MAC_QKT + MAC_Z + MAC_LINEAR + MAC_FFN)
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
    else:
        dlen = 0.5
    if kvcache_type == 'FP16':
        kvtype = 2
    elif kvcache_type == 'INT8':
        kvtype = 1
    W_QKV = (word_dims*word_dims)*3
    W_kvcache = (len*word_dims+word_dims)*2*kvtype # treat k,v as wt, read and write (4K)BW
    W_LINEAR = word_dims*word_dims
    W_FFN = (word_dims*ffn_dims)*2 + word_dims*ffn_dims
    W_NORM = word_dims
    #W_NORM = word_dims*2*2
    W_DECODER = decoder_layers * ( dlen*(W_QKV + W_FFN + W_LINEAR + 2*W_NORM)+ W_kvcache)
    #W_DECODER = decoder_layers * ( dlen*(W_QKV + W_FFN )+ W_kvcache)
    # print(W_DECODER/decoder_layers)
    W_POS_EMBD = word_dims*dlen
    W_OUT_LINEAR = vocab_size*word_dims
    return W_POS_EMBD + W_DECODER + W_OUT_LINEAR*dlen

def total_weight(in_len, out_len, wt_type='INT8', kv_type='FP16'):
    sum = 0
    for i in range(in_len, in_len+out_len):
        sum += weight(i,wt_type,kv_type)
        i += 1
    return sum

wt_size = total_weight(in_len, out_len)
wt_size_int4 = total_weight(in_len, out_len, 'INT4', 'INT8')

#print("llama parameters are dims=%d, layers=%d, heads=%d, ffn_nodes=%d"%(word_dims, decoder_layers, heads, ffn_dims))
#print("input %d words generate %d words will cost %.3fTMACs"%(in_len, out_len, mac/2**40))
#print("under %.1fTB/s BW, perf is %.1ftoken/s@FP8 and %.1ftokens@FP16"%(bw, out_len*bw/(mac/2**40), out_len*bw*0.5/(mac/2**40)))
#print("FP32 model weight size is %.2fGB, FP16 is %.2fGB, INT8 is %.2fGB"%(weight('FP32')/2**30,weight('FP16')/2**30, weight('INT8')/2**30))
#print("model weight size is %.2fGB"%(weight(0,precision)/2**30))
print("under %.2fGB/s BW, ideal perf is %.2ftoken/s@FP8, %.2ftokens/s@INT4"%(bw, out_len*bw/(wt_size/2**30),out_len*bw/(wt_size_int4/2**30)))
