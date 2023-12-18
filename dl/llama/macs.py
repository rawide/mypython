import sys

print(sys.argv, len(sys.argv))
if len(sys.argv) != 7:
    print("please input the model parameters, ./macs.py [words size] [hidden_dims] [decoder layers] [in_lens] [out_lens] [BW(TB/s)]\n")
    exit()

word_size = int(sys.argv[1])
word_dims = int(sys.argv[2])
decoder_layers = int(sys.argv[3])
heads = word_dims/128
# ffn_dims = int(sys.argv[4])
ffn_dims = int((word_dims*8/3)/256 + 0.5) * 256
in_len = int(sys.argv[4])
out_len = int(sys.argv[5])
bw = float(sys.argv[6])

def MACs(len):
    MAC_QKV = word_dims*word_dims*3 # after 1st time, generate 1token update.
    MAC_QKT = 128*len*heads # should add to previous token.
    MAC_Z = len*128*heads
    MAC_LINEAR = word_dims*word_dims
    MAC_FFN = word_dims*ffn_dims*2+ffn_dims*word_dims
    MAC_DEC = decoder_layers*(MAC_QKV + MAC_QKT + MAC_Z + MAC_LINEAR + MAC_FFN)
    MAC_OUT_LINEAR = word_size*word_dims
    return  MAC_DEC + MAC_OUT_LINEAR

def total_mac(in_len, out_len):
    sum = 0
    for i in range(1, out_len+1):
        sum += MACs(in_len+i)
        i += 1  
    return sum

mac = total_mac(in_len, out_len)


def weight(dtype):
    if dtype == 'FP32':
        dlen = 4
    elif dtype in ['FP16','BF16']:
        dlen = 2
    elif dtype == 'INT8':
        dlen = 1
    W_QKV = (word_dims*word_dims + word_dims)*3
    W_LINEAR = word_dims*word_dims + word_dims
    W_FFN = word_dims*ffn_dims*2 + word_dims + ffn_dims
    W_NORM = word_dims*2*2
    W_DECODER = decoder_layers * (W_QKV + W_LINEAR + W_FFN + W_NORM)
    # print(W_DECODER/decoder_layers)
    W_OUT_LINEAR = word_size*word_dims + word_size
    return dlen*(W_DECODER + W_OUT_LINEAR)

print("llama parameters are dims=%d, layers=%d, heads=%d, ffn_nodes=%d"%(word_dims, decoder_layers, heads, ffn_dims))
print("input %d words generate %d words will cost %.3fTMACs"%(in_len, out_len, mac/2**40))
print("under %.1fTB/s BW, perf is %.1ftoken/s@FP8 and %.1ftokens@FP16"%(bw, out_len*bw/(mac/2**40), out_len*bw*0.5/(mac/2**40)))
#print("FP32 model weight size is %.2fGB, FP16 is %.2fGB, INT8 is %.2fGB"%(weight('FP32')/2**30,weight('FP16')/2**30, weight('INT8')/2**30))