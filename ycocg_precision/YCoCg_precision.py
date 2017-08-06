#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

#channel = 'RGBL'
DEBUG_ENABLE = False  #False  #True
FILE_RECORD = True # True
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


def writeToFile(name, **matdict):
    if FILE_RECORD:
        f = open(str(name), 'w+')
        for i in index:
            f.write('------------ rampvalue ' + str(i) + ' ---------------\n')
            for j in matdict:
                matname = str(j)
                value = str((matdict[j])[i].T)
                f.write(matname + '=' + value + '\n')
        f.close()
    else:
        pass


def checkRange(matrix, type='RGB'):
    if type == 'RGB':  # RGB range is [0, 255]
        for i in range(0, 3):  # urgly
            matrix[i, 0] = min(matrix[i, 0], 255.0)
            matrix[i, 0] = max(matrix[i, 0], 0.0)
    elif type == 'YUV':  # Y rang is [16, 235], uv range is [-112, 112]
        matrix[0, 0] = min(235.0, matrix[0, 0])
        matrix[0, 0] = max(16.0, matrix[0, 0])
        matrix[1, 0] = min(112.0, matrix[1, 0])
        matrix[1, 0] = max(-112.0, matrix[1, 0])
        matrix[2, 0] = min(112.0, matrix[2, 0])
        matrix[2, 0] = max(-112.0, matrix[2, 0])
    else:
        pass


def convertImage(matrixDst, matrixSrc, csc, DstSpace='RGB', type='int'):
    for i in index:
        matrixDst.append(csc * matrixSrc[i])
        checkRange(matrixDst[i], DstSpace)
        if type == 'int':
            if DstSpace == 'RGB':
                matrixDst[i] = np.rint(matrixDst[i]).astype(np.uint8)  # RGB space is always unsigned
            else:
                matrixDst[i] = np.rint(matrixDst[i]).astype(np.int)  # YUV space has negative value


def calcuteDeltaE(matObj, matStand, deltaE):
    for i in index:
        matC = matObj[i] - matStand[i]
        result = float(sum([j * j for j in matC]))
        deltaE.append(np.sqrt(result))


def findMaxDiffComponent(matObj, matStand, maxDiff):
    for i in index:
        matC = abs(matStand[i] - matObj[i])
        print(matC.max())


redramp_RGB_raw = [np.mat([i, 0, 0]).reshape(3, 1) for i in index]
greenramp_RGB_raw = [np.mat([0, i, 0]).reshape(3, 1) for i in index]
blueramp_RGB_raw = [np.mat([0, 0, i]).reshape(3, 1) for i in index]
grayramp_RGB_raw = [np.mat([i, i, i]).reshape(3, 1) for i in index]

csc_ycocg2yuv = csc_yuv2ycocg.I
csc_rgb2yuv = csc_yuv2rgb.I
csc_pwil_new = csc_yuv2rgb * csc_ycocg2yuv

#test the dsc compression quality.
redramp_RGB_raw_dsc = []
greenramp_RGB_raw_dsc = []
blueramp_RGB_raw_dsc = []
grayramp_RGB_raw_dsc_en = []
grayramp_RGB_raw_dsc_de = []
grayramp_RGB_raw_dsc_deltaE = []
#end of test

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
redramp_RGB_ViG_deltaE = []
greenramp_RGB_ViG_deltaE = []
blueramp_RGB_ViG_deltaE = []
grayramp_RGB_ViG_deltaE = []

redramp_RGB_New = []
greenramp_RGB_New = []
blueramp_RGB_New = []
grayramp_RGB_New = []
redramp_RGB_New_deltaE = []
greenramp_RGB_New_deltaE = []
blueramp_RGB_New_deltaE = []
grayramp_RGB_New_deltaE = []
grayramp_RGB_New_MaxDiff = []

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
    convertImage(matrixDst=redramp_YCoCg, matrixSrc=redramp_YUV, csc=csc_yuv2ycocg, DstSpace='YCoCg', type='int')
    convertImage(matrixDst=redramp_RGB_New, matrixSrc=redramp_YCoCg, csc=csc_pwil_new, DstSpace='RGB', type='float')
    calcuteDeltaE(matObj=redramp_RGB_New, matStand=redramp_RGB, deltaE=redramp_RGB_New_deltaE)
    calcuteDeltaE(matObj=redramp_RGB_ViG, matStand=redramp_RGB, deltaE=redramp_RGB_ViG_deltaE)
    plt.figure(1)
    plt.xlabel('ramp step')
    plt.ylabel('DeltaE')
    plt.title('RedRamp')
    plt_redramp_new, = plt.plot(index, redramp_RGB_New_deltaE, 'ro')
    plt_redramp_vig, = plt.plot(index, redramp_RGB_ViG_deltaE, 'yo')
    plt.legend([plt_redramp_new, plt_redramp_vig], ('deltaE_New', 'deltaE_ViG'), 'best', numpoints=1)
    plt.savefig('red.jpg')
    writeToFile('redramp.txt', redramp_RGB_raw=redramp_RGB_raw, redramp_YUV=redramp_YUV, redramp_RGB=redramp_RGB, \
                redramp_YCoCg=redramp_YCoCg, redramp_RGB_ViG=redramp_RGB_ViG, redramp_RGB_New=redramp_RGB_New, redramp_RGB_New_deltaE=redramp_RGB_New_deltaE)
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
    convertImage(matrixDst=greenramp_YCoCg, matrixSrc=greenramp_YUV, csc=csc_yuv2ycocg, DstSpace='YCoCg', type='int')
    convertImage(matrixDst=greenramp_RGB_New, matrixSrc=greenramp_YCoCg, csc=csc_pwil_new, DstSpace='RGB', type='float')
    calcuteDeltaE(matObj=greenramp_RGB_New, matStand=greenramp_RGB, deltaE=greenramp_RGB_New_deltaE)
    calcuteDeltaE(matObj=greenramp_RGB_ViG, matStand=greenramp_RGB, deltaE=greenramp_RGB_ViG_deltaE)
    plt.figure(2)
    plt.xlabel('ramp step')
    plt.ylabel('DeltaE')
    plt.title('GreenRamp')
    plt_greenramp_new, = plt.plot(index, greenramp_RGB_New_deltaE, 'go')
    plt_greenramp_vig, = plt.plot(index, greenramp_RGB_ViG_deltaE, 'yo')
    plt.legend([plt_greenramp_new, plt_greenramp_vig], ('deltaE_New', 'deltaE_ViG'), 'best', numpoints=1)
    plt.savefig('green.jpg')
    writeToFile('greenramp.txt', greenramp_RGB_raw=greenramp_RGB_raw, greenramp_YUV=greenramp_YUV, greenramp_RGB=greenramp_RGB, \
                greenramp_YCoCg=greenramp_YCoCg, greenramp_RGB_ViG=greenramp_RGB_ViG, greenramp_RGB_New=greenramp_RGB_New)
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
    convertImage(matrixDst=blueramp_YCoCg, matrixSrc=blueramp_YUV, csc=csc_yuv2ycocg, DstSpace='YCoCg', type='int')
    convertImage(matrixDst=blueramp_RGB_New, matrixSrc=blueramp_YCoCg, csc=csc_pwil_new, DstSpace='RGB', type='float')
    calcuteDeltaE(matObj=blueramp_RGB_New, matStand=blueramp_RGB, deltaE=blueramp_RGB_New_deltaE)
    calcuteDeltaE(matObj=blueramp_RGB_ViG, matStand=blueramp_RGB, deltaE=blueramp_RGB_ViG_deltaE)
    plt.figure(3)
    plt.xlabel('ramp step')
    plt.ylabel('DeltaE')
    plt.title('BlueRamp')
    plt_blueramp_new, = plt.plot(index, blueramp_RGB_New_deltaE, 'bo')
    plt_blueramp_vig, = plt.plot(index, blueramp_RGB_ViG_deltaE, 'yo')
    plt.legend([plt_blueramp_new, plt_blueramp_vig], ('deltaE_New', 'deltaE_ViG'), 'best', numpoints=1)
    plt.savefig('blue.jpg')
    writeToFile('blueramp.txt', blueramp_RGB_raw=blueramp_RGB_raw, blueramp_YUV=blueramp_YUV, blueramp_RGB=blueramp_RGB, \
                blueramp_YCoCg=blueramp_YCoCg, blueramp_RGB_ViG=blueramp_RGB_ViG, blueramp_RGB_New=blueramp_RGB_New)
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
    convertImage(matrixDst=grayramp_YCoCg, matrixSrc=grayramp_YUV, csc=csc_yuv2ycocg, DstSpace='YCoCg', type='int')
    convertImage(matrixDst=grayramp_RGB_New, matrixSrc=grayramp_YCoCg, csc=csc_pwil_new, DstSpace='RGB', type='float')
    calcuteDeltaE(matObj=grayramp_RGB_New, matStand=grayramp_RGB, deltaE=grayramp_RGB_New_deltaE)
    calcuteDeltaE(matObj=grayramp_RGB_ViG, matStand=grayramp_RGB, deltaE=grayramp_RGB_ViG_deltaE)
    #findMaxDiffComponent(matObj=grayramp_RGB_New, matStand=grayramp_RGB, maxDiff=grayramp_RGB_New_MaxDiff)
    #test dsc compression quality
    #convertImage(matrixDst=grayramp_RGB_raw_dsc_en, matrixSrc=grayramp_RGB_raw, csc=csc_yuv2ycocg, DstSpace='YCoCg', type='int')
    #convertImage(matrixDst=grayramp_RGB_raw_dsc_de, matrixSrc=grayramp_RGB_raw_dsc_en, csc=csc_ycocg2yuv, DstSpace='RGB', type='float')
    #calcuteDeltaE(matObj=grayramp_RGB_raw_dsc_de, matStand=grayramp_RGB_raw, deltaE=grayramp_RGB_raw_dsc_deltaE)
    #end of test dsc
    plt.figure(4)
    plt.xlabel('ramp step')
    plt.ylabel('DeltaE')
    plt.title('GrayRamp')
    plt_grayramp_new, = plt.plot(index, grayramp_RGB_New_deltaE, 'ko')
    plt_grayramp_vig, = plt.plot(index, grayramp_RGB_ViG_deltaE, 'yo')
    #plt_grayramp_raw_dsc, = plt.plot(index, grayramp_RGB_raw_dsc_deltaE, 'co')
    #plt.legend([plt_grayramp_new, plt_grayramp_vig, plt_grayramp_raw_dsc], ('deltaE_New', 'deltaE_ViG', 'deltaE_raw_DSC'), 'best', numpoints=1)
    plt.legend([plt_grayramp_new, plt_grayramp_vig], ('deltaE_New', 'deltaE_ViG'), 'best', numpoints=1)
    plt.savefig('gray.jpg')
    writeToFile('grayramp.txt', grayramp_RGB_raw=grayramp_RGB_raw, grayramp_YUV=grayramp_YUV, grayramp_RGB=grayramp_RGB, \
                grayramp_YCoCg=grayramp_YCoCg, grayramp_RGB_ViG=grayramp_RGB_ViG, grayramp_RGB_New=grayramp_RGB_New)
    DEBUG_matrix_value('grayramp_YUV', grayramp_YUV[value], value)
    DEBUG_matrix_value('grayramp_RGB', grayramp_RGB[value], value)
    DEBUG_matrix_value('grayramp_RGB_ViG', grayramp_RGB_ViG[value], value)
    DEBUG_matrix_value('grayramp_YCoCg', grayramp_YCoCg[value], value)
    DEBUG_matrix_value('grayramp_RGB_New', grayramp_RGB_New[value], value)
    DEBUG_matrix_value('grayramp_RGB_New_deltaE', grayramp_RGB_New_deltaE[value], value)


'''
temp = []
convertImage(matrixDst=grayramp_YUV, matrixSrc=grayramp_RGB_raw, csc=csc_rgb2yuv, DstSpace='YUV', type='int')
convertImage(matrixDst=grayramp_RGB, matrixSrc=grayramp_YUV, csc=csc_yuv2rgb, DstSpace='RGB', type='float')
convertImage(matrixDst=grayramp_YCoCg, matrixSrc=grayramp_YUV, csc=csc_yuv2ycocg, DstSpace='RGB', type='int')
#convertImage(matrixDst=temp, matrixSrc=grayramp_YCoCg, csc=csc_ycocg2yuv, DstSpace='YUV', type='float')
convertImage(matrixDst=grayramp_RGB_New, matrixSrc=grayramp_YCoCg, csc=csc_pwil_new, DstSpace='RGB', type='int')
calcuteDeltaE(matObj=grayramp_RGB_New, matStand=grayramp_RGB, deltaE=grayramp_RGB_New_deltaE)
print(grayramp_RGB_New_deltaE)
'''
plt.show()


# end of code
