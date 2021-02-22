#! /usr/bin/python3
import re, sys, os

if (len(sys.argv) != 3):
    exit("invalid parameters! usage: parseMem.py [logfile] [outfile]")

if (sys.argv[1] == "--help"): 
    print("parse kernel list from log, usage: parseMem.py [logfile] [outfile]")
    exit(0)

output = sys.argv[2]

# clean output file.
if os.path.exists(output):
    os.remove(output)

with open(output, "w") as outfile:
    outfile.write("===== parse mem ======\n")


def write_to_file(str):
    with open(output, "r+") as outfile:
        if str not in outfile.readlines():
            outfile.seek(0,2)
            outfile.write(str)

with open(sys.argv[1], "r") as logfile:
    for logline in logfile.readlines():
        if 'UpdateMemoryPolicy:' in logline or 'LockMosResource:' in logline:
                write_to_file(logline)

#print ("parse Mem type done to file "+output)