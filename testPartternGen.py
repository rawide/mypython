#! /usr/bin/python3
import os, sys

def help():
    print("using: testParttern.py [format] [width] [height] [R/Y] [G/U] [B/V] [filename]")

if (len(sys.argv) != 8 and len(sys.argv) != 2):
    print("invalid parameters!")
    help()
    exit(-1)

if (sys.argv[1] == "--help" or sys.argv[1] == "-h"): 
    help()
    exit(0)

outfile = sys.argv[7]
width = int(sys.argv[2])
height = int(sys.argv[3])
R_Y = int(sys.argv[4])
G_U = int(sys.argv[5])
B_V = int(sys.argv[6])

if os.path.exists(outfile):
    os.remove(outfile)

def write_rgbp(w, h):
    with open(outfile, "wb+") as rgbp:
        for i in range(w*h):
            rgbp.write(R_Y.to_bytes(length=1,byteorder='big',signed=False))
        for i in range(w*h):
            rgbp.write(G_U.to_bytes(length=1,byteorder='big',signed=False))
        for i in range(w*h):
            rgbp.write(B_V.to_bytes(length=1,byteorder='big',signed=False))

def write_nv12(w, h):
    with open(outfile, "wb+") as nv12:
        for i in range(int(w*h)):
            nv12.write(R_Y.to_bytes(length=1,byteorder='big',signed=False))
        for i in range(int(w*h/4)):
            nv12.write(G_U.to_bytes(length=1,byteorder='big',signed=False))
            nv12.write(B_V.to_bytes(length=1,byteorder='big',signed=False))

def write_yuy2(w, h):
    with open(outfile, "wb+") as yuy2:
        for i in range(int(w*h/2)):
            yuy2.write(R_Y.to_bytes(length=1,byteorder='big',signed=False))
            yuy2.write(G_U.to_bytes(length=1,byteorder='big',signed=False))
            yuy2.write(R_Y.to_bytes(length=1,byteorder='big',signed=False))
            yuy2.write(B_V.to_bytes(length=1,byteorder='big',signed=False))

if (sys.argv[1] == "nv12"):
    write_nv12(width, height)
if (sys.argv[1] == "rgbp"):
    write_rgbp(width, height)
if (sys.argv[1] == "yuy2"):
    write_yuy2(width, height)