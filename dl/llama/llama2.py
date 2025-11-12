import sys

vocab_size = 46336

print(sys.argv, len(sys.argv))
if len(sys.argv) != 5 and len(sys.argv) != 6:
    print("please input the model parameters, ./llama2.py [wt size (7B/13B/33B/65B)] [BW(GB/s)] [in_lens] [out_lens] [cores 1/2/4(optional)]\n")
    exit()

model = sys.argv[1]
if model == "7B":
    word_dims = 4096
    decoder_layers = 32
elif model == "13B":
    word_dims = 5120
    decoder_layers = 40
elif model == "33B":
    word_dims = 6656
    decoder_layers = 60
elif model == "65B":
    word_dims = 8192
    decoder_layers = 80
else:
    print("invalid model select!")
    exit()

bw = float(sys.argv[2])
in_len = int(sys.argv[3])
out_len = int(sys.argv[4])
heads = word_dims/128
ffn_dims = int((word_dims*8/3)/256 + 0.5) * 256
X3_LLM_EFF = 0.891 # after enable wt compression, increse nearly 15% perf improve. previous is 0.76

print("hidden parameter: decoder layer = %d, FFN = %d, dim = %d"%(decoder_layers, ffn_dims, word_dims))

if len(sys.argv) == 6:
    cores = float(sys.argv[5])
    X3_CORES_EFF = 1 if cores == 1 else 0.7
    X3_MAC_EFF = 0.43 # target is 1core 43% Mac utilzation
    COMP_POWER = 8*16*16*2*2*(10**9)*cores/2 #8T MAC, 16Tops. for FP16 precision, only half comp.
    VALID_COMP_POWER = COMP_POWER*X3_CORES_EFF*X3_MAC_EFF
else:
    COMP_POWER = 0
    VALID_COMP_POWER = 0

def TTFT_MACs(len):
    MAC_QKV = len*word_dims*word_dims*3
    MAC_QKT = len*word_dims*len
    MAC_O = len*len*word_dims
    MAC_LINEAR = len*word_dims*word_dims
    MAC_FFN = len*word_dims*ffn_dims*2 + len*ffn_dims*word_dims
    MAC_DEC = decoder_layers*(MAC_QKV+MAC_QKT+MAC_O+MAC_LINEAR+MAC_FFN)
    MAC_OUT_LINEAR = vocab_size*word_dims
    return MAC_DEC + MAC_OUT_LINEAR

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
    W_kvcache = (len*word_dims+word_dims)*2*kvtype # treat k,v as wt, read and write, K/Vread + deltaK/Vwrite (4K)BW
    W_LINEAR = word_dims*word_dims
    W_FFN = (word_dims*ffn_dims)*2 + word_dims*ffn_dims
    W_NORM = word_dims
    #W_NORM = word_dims*2*2
    W_DECODER = decoder_layers * ( dlen*(W_QKV + W_FFN + W_LINEAR + 2*W_NORM)+ W_kvcache)
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

ideal_perf_INT8 = out_len*bw/(wt_size/10**9) #2**30 due to BW calc is using 10**9
ideal_perf_INT4 = out_len*bw/(wt_size_int4/10**9)
print("total size is %.2fGB for INT4, %.2fGB for INT8"%(wt_size_int4/10**9, wt_size/10**9))
print("under %.2fGB/s BW w/ X3 llm efficiency is %.2f,\nfor INT8, ideal perf is %.2ftoken/s, X3 is %.2ftoken/s,\nfor INT4, ideal perf is %.2ftokens/s, X3 is %.2ftokens/s"%
      (bw, X3_LLM_EFF, ideal_perf_INT8, ideal_perf_INT8*X3_LLM_EFF, ideal_perf_INT4, ideal_perf_INT4*X3_LLM_EFF))

if VALID_COMP_POWER != 0:
    prefill_macs = TTFT_MACs(in_len)
    ttft_comp = prefill_macs/VALID_COMP_POWER
    data_int4 = weight(in_len, 'INT4')/10**9
    ttft_ld_int4 = data_int4/bw
    data_int8 = weight(in_len, 'INT8')/10**9
    ttft_ld_int8 = data_int8/bw
    ttft_int4 = ttft_comp if ttft_comp > ttft_ld_int4 else ttft_ld_int4
    ttft_int8 = ttft_comp if ttft_comp > ttft_ld_int8 else ttft_ld_int8
    print("prefill mac is %.2fTMAC@FP16, computer power is %.2fTMAC@FP16, valid power is %.2fTMAC@FP16"%(prefill_macs/10**12, COMP_POWER/10**12, VALID_COMP_POWER/10**12))
    print("for INT8, total data is %.2fGB, ttft is %.2fs, comp time is %.2fs, ld time is %.2f"%(data_int8, ttft_int8,ttft_comp,ttft_ld_int8))
    print("for INT4, total data is %.2fGB, ttft is %.2fs, comp time is %.2fs, ld time is %.2f"%(data_int4, ttft_int4,ttft_comp,ttft_ld_int4))