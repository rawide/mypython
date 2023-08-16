import sys
import math

print(sys.argv, len(sys.argv))
if len(sys.argv) != 9:
    print("please input the model parameters, ./macs.py [words size] [hidden_dims] [decoder layers] [heads] [ffn hiden nodes] [in_lens] [out_lens] [max_len] \n")
    exit()

word_size = int(sys.argv[1])
word_dims = int(sys.argv[2])
decoder_layers = int(sys.argv[3])
heads = int(sys.argv[4])
ffn_dims = int(sys.argv[5])
in_len = int(sys.argv[6])
out_len = int(sys.argv[7])
max_len = int(sys.argv[8])

def MACs(len):
    MAC_QKV = word_dims*word_dims*len*3
    MAC_Z = word_dims*len*len*2
    MAC_LINEAR = word_dims*word_dims
    MAC_FFN = word_dims*ffn_dims*2*len
    MAC_DEC = decoder_layers* (MAC_QKV + MAC_Z + MAC_LINEAR + MAC_FFN)
    MAC_OUT_LINEAR = word_size*word_dims
    return  MAC_DEC + MAC_OUT_LINEAR

def no_opt_mac(in_len, out_len):
    sum = 0
    for i in range(0, out_len):
        sum += MACs(in_len+i)
        i += 1  
    return sum

f_in = MACs(in_len)
f_out = out_len*MACs(1)
f_no_opt = no_opt_mac(in_len, out_len)
f_max_len = out_len*MACs(max_len)

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

# print(f_in,f_out,f_no_opt)
print("input %d words generate %d words will cost %.3fTMACs, if no opt needs %.3fTMACs"%(in_len, out_len, (f_in+f_out)/math.pow(2, 40), f_no_opt/math.pow(2, 40)))
print("if using max_len, need %.2fTMAC"%(f_max_len/math.pow(2,40)))
print("FP32 model weight size is %.2fGB, FP16 is %.2fGB, INT8 is %.2fGB"%(weight('FP32')/math.pow(2, 30),weight('FP16')/math.pow(2, 30), weight('INT8')/math.pow(2, 30)))