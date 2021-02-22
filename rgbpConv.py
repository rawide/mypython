#! /usr/bin/python3
import os, sys
from PIL import Image

if (len(sys.argv) != 3 and len(sys.argv) != 2):
    exit("invalid parameters! usage: parseKernelList.py [vpkrnHeader] [logfile]")

if (sys.argv[1] == "--help" or sys.argv[1] == "-h"): 
    print("parse kernel list from log, usage: parseKernelList.py [vpkrnHeader] [logfile]")
    exit(0)

dst_img = sys.argv[2]
if os.path.exist(dst_img):
    os.remove(dst_img)

src_img = array(Image.open(sys.argv[1]))
print(src_img.sharp)
""" with open(output, "a+") as dst_img:
    dst_img.write(src_img)
 """
