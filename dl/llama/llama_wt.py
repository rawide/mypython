import sys

print(sys.argv, len(sys.argv))
if len(sys.argv) != 5:
    print("please input the model parameters, ./macs.py [model:llama2 or llama3] [words size] [hidden_dims] [decoder layers] [GQA num]\n")
    exit()

if sys.argv[1] == 'llama2':
    vocab_size = 32016
elif sys.argv[1] == 'llama3':
    vocab_size = 128256
else :
    print("please input the model, ./macs.py [model:llama2 or llama3] [words size] [hidden_dims] [decoder layers] [GQA num]\n")
    exit()

dims = int(sys.argv[2])
decoder_layers = int(sys.argv[3])
gqa_num = int(sys.argv[4])
ffn_dims = int((dims*8/3)/256 + 0.5) * 256

size_tokenizer = vocab_size*dims
size_norm = dims*2
size_wq = dims*dims + dims
size_wkv = 2*(dims*dims/gqa_num + dims/gqa_num)
size_linear = dims*dims + dims
size_ffn = 2*(dims*ffn_dims+ffn_dims) + ffn_dims*dims+dims
size_dec = decoder_layers*(size_wq+size_wkv+size_linear+size_ffn+size_norm)
size_wt = size_tokenizer+size_dec + dims*vocab_size + vocab_size

print("weight cnt is %.2fB, total %.2fGB"%(size_wt/10**9, size_wt/2**30))