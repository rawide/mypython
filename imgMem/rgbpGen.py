#! /usr/bin/python3
import os, sys

if (len(sys.argv) != 7 and len(sys.argv) != 2):
    exit("invalid parameters! usage: rgbGen.py [width] [heigt] [R] [G] [B] [filename]")

if (sys.argv[1] == "--help" or sys.argv[1] == "-h"): 
    print("invalid parameters! usage: rgbGen.py [width] [heigt] [R] [G] [B] [filename]")
    exit(0)

outfile = sys.argv[6]
width = int(sys.argv[1])
height = int(sys.argv[2])
size = width * height
R = int(sys.argv[3])
G = int(sys.argv[4])
B = int(sys.argv[5])

if os.path.exists(outfile):
    os.remove(outfile)

with open(outfile, "wb+") as rgbp:
    for i in range(size):
        rgbp.write(R.to_bytes(length=1,byteorder='big',signed=False))
    for i in range(size):
        rgbp.write(G.to_bytes(length=1,byteorder='big',signed=False))
    for i in range(size):
        rgbp.write(B.to_bytes(length=1,byteorder='big',signed=False))