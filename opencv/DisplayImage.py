#! /usr/bin/python2.7
import cv2 as cv
import sys

print("opencv version is "+cv.__version__)
if (len(sys.argv) != 2):
    sys.exit("usage: ./DisplayImage.py [filename]")
print("open file:"+sys.argv[1])
img = cv.imread(sys.argv[1])
if img is None:
    sys.exit("could not open image ")

cv.imshow("show Image", img)
cv.waitKey(0)
# cv.destroyAllWindows()

img_scale = cv.resize(img, (224, 224))
cv.imshow("resize Image", img_scale)
cv.waitKey(0)
cv.destroyAllWindows()