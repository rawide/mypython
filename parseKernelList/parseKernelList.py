#! /usr/bin/python3
import re, sys, os

if (len(sys.argv) != 3):
    exit("invalid parameters! usage: parseKernelList.py [vpkrnheader.h] [logfile]")

if (sys.argv[1] == "--help"): 
    print("parse kernel list from log, usage: parseKernelList.py [vpkrnheader.h] [logfile]")
    exit(0)

output = "kernellist.txt"

# find kernel rule list in heard file
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

# clean output file.
if os.path.exists(output):
    os.remove(output)

# find kernel list num in captured log file.
with open(sys.argv[2], "r") as LogFile:
    for logline in LogFile.readlines():
        if 'Kdll_AddKernelList:4850:' in logline:
            #print(logline.find(':4850: ', 10, len(logline))), find kernel list start location,
            find_krn_num = logline[logline.find(':4850: ', 10, len(logline))+7:len(logline)-2] #pick up the kernel list
            #print(find_krn_num)
            write_to_file(find_krn_name(find_krn_num))

print ("find kernel list and output to file "+output)