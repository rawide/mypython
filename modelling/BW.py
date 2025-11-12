import sys

print(sys.argv, len(sys.argv))
if len(sys.argv) != 8:
    print("please input the model parameters, ./macs.py [DDR clk Ghz] [xpu clk Ghz] [ddr latency cycle] [xpu latency cycle] [xpu outstanding] [read burst lenght] [DDR BW]\n")
    exit()

clk_ddr = float(sys.argv[1])
clk_xpu = float(sys.argv[2])
latency_ddr = int(sys.argv[3])
latency_xpu = int(sys.argv[4])
ot = int(sys.argv[5])
brust_lenght = int(sys.argv[6])
ddr_bw = int(sys.argv[7])

tc_ddr = int(round(1.0/clk_ddr,1)*10)
tc_xpu = int(round(1.0/clk_xpu,1)*10)

cycle_ddr = 0
cycle_xpu = 0

def ddr_cycle(i):
    if i%tc_ddr == 0:
        global cycle_ddr += 1
        return True
    else:
        return False

def xpu_cycle(i):
    if i%tc_xpu == 0:
        global cycle_xpu += 1
        return True
    else:
        return False

Timecnt = 10**3

def run():
    i = 1
    while (i<Timecnt):
        ddr_cycle(i)
        xpu_cycle(i)
        i += 1

run()

# print("xpu and ddr cycle time is %d, %d"%(tc_xpu,tc_ddr))
print("xpu and ddr cycle is %d, %d"%(cycle_xpu, cycle_ddr))