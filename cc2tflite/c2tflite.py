import re
import sys

print(sys.argv, len(sys.argv))
if len(sys.argv) != 2:
    print("usage: python %s [input_c_file]"%(sys.argv[0]))
    exit()

# 读取 .cc 文件
with open(sys.argv[1], "r") as f:
    text = f.read()

# 提取模型数据数组（十六进制数）
hex_values = re.findall(r'0x[0-9a-fA-F]{2}', text)
byte_array = bytes(int(h, 16) for h in hex_values)

# 写出为 .tflite 文件
with open(sys.argv[1]+".tflite", "wb") as f:
    f.write(byte_array)

print("模型已保存为 %s.tflite"%(sys.argv[1]+"tflite"))

