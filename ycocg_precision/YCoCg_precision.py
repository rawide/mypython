#! /usr/bin/python
import matplotlib.pyplot as mplt
import numpy as np

#channel = 'RGBL'
DEBUG_ENABLE = True
index = range(0, 256)
csc_yuv2rgb = np.mat('1.1644   0.0      1.6787; \
                      1.1644  -0.1873  -0.6504; \
                      1.1644   2.1418   0.0')
csc_yuv2ycocg = np.mat('1   0.5  -0.5; \
                        1     0   0.5; \
                        1  -0.5  -0.5')


# parameters and functions define
def DEBUG_matrix_value(str, matrix, value):
    if DEBUG_ENABLE:
        print('matrix %s[%d] value is' % (str, value))
        print(matrix)
    else:
        pass


def checkRange(matrix, type='RGB'):
    if type == 'RGB':  # RGB range is [0, 255]
        for i in range(0, 3):  # urgly
            matrix[i, 0] = min(matrix[i, 0], 255.0)
            matrix[i, 0] = max(matrix[i, 0], 0.0)
    else:  # Y rang is [16, 235], uv range is [-112, 112]
        matrix[0, 0] = min(235.0, matrix[0, 0])
        matrix[0, 0] = max(16.0, matrix[0, 0])
        matrix[1, 0] = min(112.0, matrix[1, 0])
        matrix[1, 0] = max(-112.0, matrix[1, 0])
        matrix[2, 0] = min(112.0, matrix[2, 0])
        matrix[2, 0] = max(-112.0, matrix[2, 0])


def convertImage(matrixDst, matrixSrc, csc, DstSpace='RGB', type='int'):
    for i in index:
        matrixDst.append(csc * matrixSrc[i])
        checkRange(matrixDst[i], DstSpace)
        if type == 'int':
            if DstSpace == 'RGB':
                matrixDst[i] = np.rint(matrixDst[i]).astype(np.uint8)  # RGB space is always unsigned
            else:
                matrixDst[i] = np.rint(matrixDst[i]).astype(np.int)  # YUV space has negative value


def calcuteDeltaE(matA, matB, deltaE):
    for i in index:
        matC = matA[i] - matB[i]
        result = float(sum([j * j for j in matC]))
        deltaE.append(np.sqrt(result))


def findMaxDiff(matObj, matStand, maxDiff):
    for i in index:
        


redramp_RGB_raw = [np.mat([i, 0, 0]).reshape(3, 1) for i in index]
greenramp_RGB_raw = [np.mat([0, i, 0]).reshape(3, 1) for i in index]
blueramp_RGB_raw = [np.mat([0, 0, i]).reshape(3, 1) for i in index]
grayramp_RGB_raw = [np.mat([i, i, i]).reshape(3, 1) for i in index]

csc_ycocg2yuv = csc_yuv2ycocg.I
csc_rgb2yuv = csc_yuv2rgb.I
redramp_YUV = []
greenramp_YUV = []
blueramp_YUV = []
grayramp_YUV = []

redramp_YCoCg = []
greenramp_YCoCg = []
blueramp_YCoCg = []
grayramp_YCoCg = []

redramp_RGB = []
greenramp_RGB = []
blueramp_RGB = []
grayramp_RGB = []

redramp_RGB_ViG = []
greenramp_RGB_ViG = []
blueramp_RGB_ViG = []
grayramp_RGB_ViG = []

redramp_RGB_New = []
greenramp_RGB_New = []
blueramp_RGB_New = []
grayramp_RGB_New = []
redramp_RGB_New_deltaE = []
greenramp_RGB_New_deltaE = []
blueramp_RGB_New_deltaE = []
grayramp_RGB_New_deltaE = []
grayramp_RGB_New_MaxDiff = {}

# useage introduce
print('''usage:
R = red color; G = green color; B = blue color; L = luma/gray color
for example: RG or B or GL''')
if DEBUG_ENABLE:
    channel, value = raw_input('which channel you want to simulate and pick a value to see?\n').split()
    value = int(value)
else:
    channel = raw_input('which channel you want to simulate?\n')
    value = 255

# start of code
if 'R' in channel:
    convertImage(matrixDst=redramp_YUV, matrixSrc=redramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
    convertImage(matrixDst=redramp_RGB, matrixSrc=redramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
    convertImage(matrixDst=redramp_RGB_ViG, matrixSrc=redramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
    convertImage(matrixDst=redramp_YCoCg, matrixSrc=redramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
    convertImage(matrixDst=redramp_RGB_New, matrixSrc=redramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
    calcuteDeltaE(matA=redramp_RGB_New, matB=redramp_RGB, deltaE=redramp_RGB_New_deltaE)
    DEBUG_matrix_value('redramp_YUV', redramp_YUV[value], value)
    DEBUG_matrix_value('redramp_RGB', redramp_RGB[value], value)
    DEBUG_matrix_value('redramp_RGB_ViG', redramp_RGB_ViG[value], value)
    DEBUG_matrix_value('redramp_YCoCg', redramp_YCoCg[value], value)
    DEBUG_matrix_value('redramp_RGB_New', redramp_RGB_New[value], value)
    DEBUG_matrix_value('redramp_RGB_New_deltaE', redramp_RGB_New_deltaE[value], value)

if 'G' in channel:
    convertImage(matrixDst=greenramp_YUV, matrixSrc=greenramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
    convertImage(matrixDst=greenramp_RGB, matrixSrc=greenramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
    convertImage(matrixDst=greenramp_RGB_ViG, matrixSrc=greenramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
    convertImage(matrixDst=greenramp_YCoCg, matrixSrc=greenramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
    convertImage(matrixDst=greenramp_RGB_New, matrixSrc=greenramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
    calcuteDeltaE(matA=greenramp_RGB_New, matB=greenramp_RGB, deltaE=greenramp_RGB_New_deltaE)
    DEBUG_matrix_value('greenramp_YUV', greenramp_YUV[value], value)
    DEBUG_matrix_value('greenramp_RGB', greenramp_RGB[value], value)
    DEBUG_matrix_value('greenramp_RGB_ViG', greenramp_RGB_ViG[value], value)
    DEBUG_matrix_value('greenramp_YCoCg', greenramp_YCoCg[value], value)
    DEBUG_matrix_value('greenramp_RGB_New', greenramp_RGB_New[value], value)
    DEBUG_matrix_value('greenramp_RGB_New_deltaE', greenramp_RGB_New_deltaE[value], value)

if 'B' in channel:
    convertImage(matrixDst=blueramp_YUV, matrixSrc=blueramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
    convertImage(matrixDst=blueramp_RGB, matrixSrc=blueramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
    convertImage(matrixDst=blueramp_RGB_ViG, matrixSrc=blueramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
    convertImage(matrixDst=blueramp_YCoCg, matrixSrc=blueramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
    convertImage(matrixDst=blueramp_RGB_New, matrixSrc=blueramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
    calcuteDeltaE(matA=blueramp_RGB_New, matB=blueramp_RGB, deltaE=blueramp_RGB_New_deltaE)
    DEBUG_matrix_value('blueramp_YUV', blueramp_YUV[value], value)
    DEBUG_matrix_value('blueramp_RGB', blueramp_RGB[value], value)
    DEBUG_matrix_value('blueramp_RGB_ViG', blueramp_RGB_ViG[value], value)
    DEBUG_matrix_value('blueramp_YCoCg', blueramp_YCoCg[value], value)
    DEBUG_matrix_value('blueramp_RGB_New', blueramp_RGB_New[value], value)
    DEBUG_matrix_value('bluenramp_RGB_New_deltaE', blueramp_RGB_New_deltaE[value], value)

if 'L' in channel:
    convertImage(matrixDst=grayramp_YUV, matrixSrc=grayramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
    convertImage(matrixDst=grayramp_RGB, matrixSrc=grayramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
    convertImage(matrixDst=grayramp_RGB_ViG, matrixSrc=grayramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
    convertImage(matrixDst=grayramp_YCoCg, matrixSrc=grayramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
    convertImage(matrixDst=grayramp_RGB_New, matrixSrc=grayramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
    calcuteDeltaE(matA=grayramp_RGB_New, matB=grayramp_RGB, deltaE=grayramp_RGB_New_deltaE)
    DEBUG_matrix_value('grayramp_YUV', grayramp_YUV[value], value)
    DEBUG_matrix_value('grayramp_RGB', grayramp_RGB[value], value)
    DEBUG_matrix_value('grayramp_RGB_ViG', grayramp_RGB_ViG[value], value)
    DEBUG_matrix_value('grayramp_YCoCg', grayramp_YCoCg[value], value)
    DEBUG_matrix_value('grayramp_RGB_New', grayramp_RGB_New[value], value)
    DEBUG_matrix_value('grayramp_RGB_New_deltaE', grayramp_RGB_New_deltaE[value], value)



'''
convertImage(matrixDst=redramp_YUV, matrixSrc=redramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
convertImage(matrixDst=greenramp_YUV, matrixSrc=greenramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
convertImage(matrixDst=blueramp_YUV, matrixSrc=blueramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
convertImage(matrixDst=grayramp_YUV, matrixSrc=grayramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
DEBUG_matrix_value('redramp_YUV[255]', redramp_YUV[255])
DEBUG_matrix_value('grayramp_YUV[255]', grayramp_YUV[255])

convertImage(matrixDst=redramp_RGB, matrixSrc=redramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
convertImage(matrixDst=greenramp_RGB, matrixSrc=greenramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
convertImage(matrixDst=blueramp_RGB, matrixSrc=blueramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
convertImage(matrixDst=grayramp_RGB, matrixSrc=grayramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
#DEBUG_matrix_value('redramp_RGB[255]', redramp_RGB[255])
DEBUG_matrix_value('grayramp_RGB[255]', grayramp_RGB[255])

convertImage(matrixDst=redramp_RGB_ViG, matrixSrc=redramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
convertImage(matrixDst=greenramp_RGB_ViG, matrixSrc=greenramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
convertImage(matrixDst=blueramp_RGB_ViG, matrixSrc=blueramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
convertImage(matrixDst=grayramp_RGB_ViG, matrixSrc=grayramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='int')
#DEBUG_matrix_value('redramp_RGB_ViG[255]', redramp_RGB_ViG[255])
DEBUG_matrix_value('grayramp_RGB_ViG[255]', grayramp_RGB_ViG[255])

#YCoCg is also 8bit
convertImage(matrixDst=redramp_YCoCg, matrixSrc=redramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
convertImage(matrixDst=greenramp_YCoCg, matrixSrc=greenramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
convertImage(matrixDst=blueramp_YCoCg, matrixSrc=blueramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
convertImage(matrixDst=grayramp_YCoCg, matrixSrc=grayramp_YUV, csc=csc_yuv2ycocg, DstSpace='YUV', type='int')
#DEBUG_matrix_value('redramp_YCoCg[255]', redramp_YCoCg[255])
DEBUG_matrix_value('grayramp_YCoCg[255]', grayramp_YCoCg[255])

#redramp_YUV_restore = []
#convertImage(matrixDst=redramp_YUV_restore, matrixSrc=redramp_YCoCg, csc=csc_ycocg2yuv, DstSpace='YUV', type='float')
#DEBUG_matrix_value('resetore redramp_YUV_restore[255]', redramp_YUV_restore[255])
#convertImage(matrixDst=redramp_RGB_New, matrixSrc=redramp_YUV_restore, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
convertImage(matrixDst=redramp_RGB_New, matrixSrc=redramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
convertImage(matrixDst=greenramp_RGB_New, matrixSrc=greenramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
convertImage(matrixDst=blueramp_RGB_New, matrixSrc=blueramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
convertImage(matrixDst=grayramp_RGB_New, matrixSrc=grayramp_YCoCg, csc=(csc_yuv2rgb * csc_ycocg2yuv), DstSpace='RGB', type='float')
#DEBUG_matrix_value('redramp_RGB_New[255]', redramp_RGB_New[255])
DEBUG_matrix_value('grayramp_RGB_New[255]', grayramp_RGB_New[255])
'''

'''
redramp_YUV = [np.rint(csc_rgb2yuv * redramp_RGB_raw[i]).astype(np.int8) for i in index]
greenramp_YUV = [np.rint(csc_rgb2yuv * greenramp_RGB_raw[i]).astype(np.int8) for i in index]
blueramp_YUV = [np.rint(csc_rgb2yuv * blueramp_RGB_raw[i]).astype(np.int8) for i in index]
grayramp_YUV = [np.rint(csc_rgb2yuv * grayramp_RGB_raw[i]).astype(np.int8) for i in index]
DEBUG_matrix_value('redramp_YUV[255]', redramp_YUV[255])

redramp_RGB = [csc_yuv2rgb * redramp_YUV[i] for i in index]
greenramp_RGB = [csc_yuv2rgb * redramp_YUV[i] for i in index]
blueramp_RGB = [csc_yuv2rgb * redramp_YUV[i] for i in index]
grayramp_RGB = [csc_yuv2rgb * redramp_YUV[i] for i in index]
checkRGBrange(redramp_RGB)
checkRGBrange(greenramp_RGB)
checkRGBrange(blueramp_RGB)
checkRGBrange(grayramp_RGB)
DEBUG_matrix_value('redramp_RGB[255]', redramp_RGB[255])

redramp_RGB_ViG = [np.rint(redramp_RGB[i]).astype(np.uint8) for i in index]
greenramp_RGB_ViG = [np.rint(greenramp_RGB[i].astype(np.uint8)) for i in index]
blueramp_RGB_ViG = [np.rint(blueramp_RGB[i].astype(np.uint8)) for i in index]
grayramp_RGB_ViG = [np.rint(grayramp_RGB[i]).astype(np.uint8) for i in index]
DEBUG_matrix_value('redramp_RGB_ViG[255]', redramp_RGB_ViG[255])

redramp_YCoCg = [np.rint(csc_yuv2ycocg * redramp_YUV[i]).astype(np.int8) for i in index]
greenramp_YCoCg = [np.rint(csc_yuv2ycocg * greenramp_YUV[i]).astype(np.int8) for i in index]
blueramp_YCoCg = [np.rint(csc_yuv2ycocg * blueramp_YUV[i]).astype(np.int8) for i in index]
grayramp_YCoCg = [np.rint(csc_yuv2ycocg * grayramp_YUV[i]).astype(np.int8) for i in index]
DEBUG_matrix_value('redramp_YCoCg[255]', redramp_YCoCg[255])

redramp_RGB_New = [np.rint(csc_yuv2rgb * (csc_ycocg2yuv * redramp_YCoCg[i])).astype(np.int8) for i in index]
greenramp_RGB_New = [np.rint(csc_yuv2rgb * (csc_ycocg2yuv * greenramp_YCoCg[i])).astype(np.int8) for i in index]
blueramp_RGB_New = [np.rint(csc_yuv2rgb * (csc_ycocg2yuv * blueramp_YCoCg[i])).astype(np.int8) for i in index]
grayramp_RGB_New = [np.rint(csc_yuv2rgb * (csc_ycocg2yuv * grayramp_YCoCg[i])).astype(np.int8) for i in index]
DEBUG_matrix_value('redramp_RGB_New[255]', redramp_RGB_New[255])
'''



# end of code
