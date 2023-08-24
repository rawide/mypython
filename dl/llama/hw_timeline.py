from matplotlib import pyplot as plt
import numpy as np
import math

print("simulation LLAMA 7B model running time memory footprint.")
print("model parameter is vocab_size = 32000, d_model = 4096, decoder_layers = 32, num_heads = 32 \
ffn_nodes = 11008, input_len = 8, output_len = 256")

# print(sys.argv, len(sys.argv))
# if len(sys.argv) != 9:
#     print("please input the model parameters, ./macs.py [words size] [hidden_dims] [decoder layers] [heads] [ffn hiden nodes] [in_lens] [out_lens] [max_len] \n")
#     exit()

# define parameters
vocab_size = 32000
d_model = 4096
num_heads = 32
num_decoder = 32
ffn_dims = 11008
input_len = 8
output_len = 256
max_len = 1024
d_k = d_model // num_heads
norm_timing = 0.0475*1000*1000 # norm timing cost, 0.0475ms for d_model vector norm.
hw_MAC = 1024 # hw macs
clk_MAC = 0.050 # 50Mhz, 0.05Ghz
lut_timing = 15*clk_MAC # 200 clk for once SRAM, ns.
acc = 'INT8' # FP 16
data_size = 1 if acc == 'INT8' else 2
softmaxLUT = 0.25 # 0.25MB for softmax lut, mapping FP16 value

def CALC_POWER():
    global hw_MAC, clk_MAC
    if hw_MAC >= d_model:
        cal_power = (hw_MAC // d_model) * d_model * clk_MAC
    else:
        cal_power = (d_model/round(d_model/hw_MAC + 0.5))*clk_MAC
    return cal_power

# define each time computation cost
def MAC_Q(len):
    return d_model*d_k*len
MAC_K = MAC_V = MAC_Q

def MAC_Attn(len):
    return d_k*len*len

def MAC_Z(len):
    return len*d_k*len

def MAC_Linear(len):
    return d_model*d_model*len

def MAC_FFN_L1(len):
    return d_model*ffn_dims*len

def MAC_FFN_L2(len):
    return ffn_dims*d_model*len

def MAC_DEC_BLOCK(len):
    return num_heads*(3*MAC_Q(len)+MAC_Attn(len))+MAC_Linear(len)+MAC_FFN_L1(len)+MAC_FFN_L2(len)

def TIME_Q(len):
    return MAC_Q(len)/CALC_POWER()
TIME_K = TIME_V = TIME_Q

def TIME_Attn(len):
    return MAC_Attn(len)/CALC_POWER()

def TIME_Attn_softmax(len): # non-linear calculation, LUT
    return (len*lut_timing)*len

def TIME_Z(len):
    return MAC_Z(len)/CALC_POWER()

def TIME_Linear(len):
    return MAC_Linear(len)/CALC_POWER()

def TIME_FFN_L1(len):
    return MAC_FFN_L1(len)/CALC_POWER()

def TIME_FFN_L2(len):
    return MAC_FFN_L2(len)/CALC_POWER()

def TIME_NORM(len):
    return len*norm_timing

def TIME_FFN(len):
    return TIME_FFN_L1(len)+TIME_FFN_L2(len)

def TIME_MHA(len):
    return num_heads*(3*TIME_Q(len)+TIME_Attn(len)+TIME_Attn_softmax(len)+TIME_Z(len))

def TIME_Dec(len):
    return TIME_MHA(len)+TIME_Linear(len)+TIME_NORM(len)+TIME_FFN(len)+TIME_NORM(len)

def MAC_Last_Linear(): # only last token needs to map as words
    return d_model*vocab_size

def TIME_Last_Linear():
    return MAC_Last_Linear()/CALC_POWER()

def TIME_Last_softmax():
    return vocab_size*lut_timing

def TIME_ONE_TOKEN(len):
    return num_decoder*TIME_Dec(len) + TIME_Last_Linear() + TIME_Last_softmax()

def MAC_TOTAL(len):
    return MAC_Last_Linear()+num_decoder*MAC_DEC_BLOCK(len)

# for each token generate, input is (seq_len，d_model), here seq_len = input_len + output_len
def TOTAL_TIME_COST():
    total_time = 0
    total_mac = 0
    # for i in range(0, output_len):
    #     total_time += TIME_ONE_TOKEN(input_len + i)
    #     total_mac += MAC_TOTAL(input_len+i)
    # total_time = output_len*TIME_ONE_TOKEN(max_len)
    # total_mac = output_len*MAC_TOTAL(max_len)
    total_time = output_len*TIME_ONE_TOKEN(input_len+output_len)
    total_mac = output_len*MAC_TOTAL(input_len+output_len)
    print("input %d and output %d token, total macs is %.2fT, w/ %d hw Macs needs %.2fs" % (input_len, output_len, total_mac/pow(2,40), hw_MAC, total_time/pow(10,9)))
    return total_time/pow(10,9)

TOTAL_TIME_COST()

'''
hw pipeline
for each attention, the cal includes heads internal, Q, K, attn, softmax, V, Z
mha output concat and linear Wo, NORM, FFN_L1, FFN_L2, NORM calc
'''

# output Q, K, attn, softmax(attn), V, Z time
def TIMECOST_HEAD(len, unit='us'):
    if unit == 'us':
        scaler = 1000
    elif unit == 'ms':
        scaler = 1000*1000
    return [TIME_Q(len)/scaler, TIME_K(len)/scaler, TIME_Attn(len)/scaler, TIME_Attn_softmax(len)/scaler,TIME_V(len)/scaler, TIME_Z(len)/scaler]

# output linear, NORM time
def TIMECOST_MHA(len, unit='us'):
    if unit == 'us':
        scaler = 1000
    elif unit == 'ms':
        scaler = 1000*1000
    return [TIME_Linear(len)/scaler, TIME_NORM(len)/scaler]

# output FFN_L1, FFN_L2, NORM time
def TIMECOST_DEC(len, unit='us'):
    if unit == 'us':
        scaler = 1000
    elif unit == 'ms':
        scaler = 1000*1000
    return [TIME_FFN_L1(len)/scaler, TIME_FFN_L2(len)/scaler, TIME_NORM(len)/scaler]

# each head calculate memory cost, Q, K, attn, softmax(attn), V, Z
def MEMCOST_HEAD(len, unit='MB'):
    if unit == 'MB':
        scaler = 1024*1024
    elif unit == 'KB':
        scaler = 1024
    base = len*d_model/scaler # store concat Z
    Wk = Wv = W_q = d_model*d_k/scaler
    Z = Q = K = V = len*d_k/scaler
    attn = len*len/scaler
    mcalcK = mcalcV = mcalcQ = (len*d_model + W_q)/scaler
    return [mcalcQ+base, mcalcK+Q+base, Q+K+attn+base, attn+base, attn+mcalcV+base, attn+V+Z+base]

# define mha output memory cost linear, norm
def MEMCOST_MHA(len, unit='MB'):
    if unit == 'MB':
        scaler = 1024*1024
    elif unit == 'KB':
        scaler = 1024
    base = len*d_model/scaler # store concat Z
    W_linear = (d_model*d_model+d_model)/scaler
    W_norm = 2*2*d_model/scaler
    return [base+W_linear+base, base+W_norm+base]

# define decoder memory cost of FFN_L1, FFN_L2, norm
def MEMCOST_DEC(len, unit='MB'):
    if unit == 'MB':
        scaler = 1024*1024
    elif unit == 'GB':
        scaler = 1024*1024*1024
    base = base = len*d_model/scaler # store Wo
    W_linear1 = (ffn_dims*d_model+ffn_dims)/scaler
    O_linear1 = len*ffn_dims/scaler
    W_linear2 = (d_model*ffn_dims+d_model)/scaler
    O_linear2 = len*d_model/scaler
    W_norm = 2*2*d_model/scaler
    return [base+W_linear1+O_linear1, O_linear1+W_linear2+O_linear2, O_linear2+W_norm+base]

def draw_mem_time():
    plt.figure()
    plt.subplot(4,1,1)
    timecost_head = TIMECOST_HEAD(input_len+output_len, 'ms')
    memcost_head = MEMCOST_HEAD(input_len+output_len)
    print(f"[Q, K, attn, softmax(attn), V, Z] timecost={timecost_head}, memcost={memcost_head}")
    t_head_left_boundaries = np.cumsum(timecost_head) - timecost_head
    # colors = ['green', 'blue']
    # plt.bar(x_left_boundaries, memcost_head, width=timecost_head, color=colors, align='edge')
    plt.bar(t_head_left_boundaries, memcost_head, width=timecost_head, edgecolor='black', align='edge')
    # print(t_head_left_boundaries)
    plt.title("head (Q,K,attn,softmax,V,Z)")
    plt.xlabel("time ms")
    plt.ylabel("memory MB")

    plt.subplot(4,1,2)
    timecost_mha = TIMECOST_MHA(input_len+output_len, 'ms')
    memcost_mha = MEMCOST_MHA(input_len+output_len)
    # print(timecost_mha, memcost_mha)
    t_mha_left_boundaries = np.cumsum(timecost_mha) - timecost_mha
    plt.bar(t_mha_left_boundaries, memcost_mha, width=timecost_mha, edgecolor='black', align='edge')
    plt.title("attention (linear,norm)")
    plt.xlabel("time ms")
    plt.ylabel("memory MB")

    plt.subplot(4,1,3)
    timecost_dec = TIMECOST_DEC(input_len+output_len,'ms')
    memcost_dec = MEMCOST_DEC(input_len+output_len)
    # print(f"[]{timecost_dec}, {memcost_dec}")
    t_dec_left_boundaries = np.cumsum(timecost_dec) - timecost_dec
    plt.bar(t_dec_left_boundaries, memcost_dec, width=timecost_dec, edgecolor='black', align='edge')
    plt.title("dec (FFN_L1, FFN_L2,norm)")
    plt.xlabel("time ms")
    plt.ylabel("memory MB")

    plt.subplot(4,1,4)
    dec_layer_time = []
    dec_layer_mem = []
    for i in range(0, num_heads):
        dec_layer_time += timecost_head
        dec_layer_mem += memcost_head
    dec_layer_time += (timecost_mha + timecost_dec)
    # dec_layer_time += timecost_dec
    dec_layer_mem += (memcost_mha + memcost_dec)
    # dec_layer_mem += memcost_dec
    # print(dec_layer_time, dec_layer_mem)
    t_dec_layer_left_boundaries = np.cumsum(dec_layer_time) - dec_layer_time
    # print(t_dec_layer_left_boundaries)
    plt.bar(t_dec_layer_left_boundaries, dec_layer_mem, width=dec_layer_time, edgecolor='black', align='edge')
    plt.title("dec layer")
    plt.xlabel("time ms")
    plt.ylabel("memory MB")

    plt.tight_layout()
    # plt.show()

def draw_mac_time():
    '''
    prepare the max parallel pipeline, assuem the computation is unlimit, the max parallel pipeline is
    ============================================
       |thread1|thread2|thread3|thread4|thread5|thread6|thread7|...
       |-------|-------|-------|-------|-------|-------|-------|...
    t0 |   Q   |   K   |   V   |       |       |       |       |...
    t1 |  attn |softmax|   Z   | Linear| norm  |FFN_L1 | FFNL2 |...  // line based delay
    '''

def draw_mac_num_time():
    global hw_MAC
    x = range(512, 5000, 512)
    y = []
    for i in x:
        hw_MAC = i
        y += [TOTAL_TIME_COST()]
    plt.figure()
    x_labels = [str(val) for val in x]
    plt.bar(x_labels, y, color='blue')
    plt.title("hw MAC num perf")
    plt.xlabel("hw MACs")
    plt.ylabel("performance(s)")
    # plt.show()

draw_mem_time()
# draw_mac_num_time()
plt.show()