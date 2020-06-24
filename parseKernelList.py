#! /usr/bin/python
import re, sys, os

if (len(sys.argv) != 3 and len(sys.argv) != 2):
    exit("invalid parameters! usage: parseKernelList.py [vpkrnHeader] [logfile]")

if (sys.argv[1] == "--help"): 
    print("parse kernel list from log, usage: parseKernelList.py [vpkrnHeader] [logfile]")
    exit(0)

output = "kernellist.txt"

def find_krn_name(num):
    with open(sys.argv[1], "r") as KernelHeaderFile:
        for headerline in KernelHeaderFile.readlines():
            if 'define IDR_VP' in headerline:
                headerKrnNum = re.split(' ', headerline)
                if int(num.strip()) == int(headerKrnNum[2].strip()):
                    find_krn_name = re.sub('#define ','',headerline)
                    #print(find_krn_name)
                    return find_krn_name

def write_to_file(str):
    with open(output, "a+") as outfile:
        outfile.write(str)

# clean kernelList file.
if os.path.exists(output):
    os.remove(output)

with open(sys.argv[2], "r") as LogFile:
    for logline in LogFile.readlines():
        if 'Component kernels [' in logline:
            #print(logline.find(']: ', 10, len(logline)))
            find_krn_num = logline[logline.find(']: ', 10, len(logline))+2:]
            #print(find_krn_num)
            write_to_file(find_krn_name(find_krn_num))

print ("find kernel list and output to file "+output)