#! /usr/bin/python
import matplotlib.pyplot as plt

x = list(range(16, 218))
csc = 1.1644
y = [0.5, 0.25]
yy = [2, 4]
#f = open("all_data.txt", "w+")
print('for yuv data, the diff from 16 to 218 is following\n')
for x in x:
    outx = x * csc
    outx_pre = int(outx+0.5)
    outx_post_half = (int(x*y[0]+0.5))*yy[0]*csc
    outx_post_quar = (int(x*y[1]+0.5))*yy[1]*csc
    acc_precent_half = (1-abs(outx-outx_post_half)/outx)*100
    acc_precent_quar = (1-abs(outx-outx_post_quar)/outx)*100
    acc_vig = (1-abs(outx-outx_pre)/outx)*100
    improving_half = acc_precent_half - acc_vig
    improving_quar = acc_precent_quar - acc_vig

    print('''x = %d, real_value = %.4f
    ViG_CSC2RGB = %d, acc_vig = %.2f%%
    half_YCoCg = %.4f, acc_half = %.2f%%, improving = %.2f%%
    quar_YCoCg = %.4f, acc_quar = %.2f%%, improving = %.2f%%\n\t
    ''' % (x, outx, outx_pre, acc_vig, outx_post_half, acc_precent_half, \
    improving_half, outx_post_quar, acc_precent_quar, improving_quar))


fullrange = range(16, 218)
lowfq = range(16, 128)
highfq = range(128, 218)

vig_rgb = [int(x*csc+0.5)/(x*csc) for x in fullrange]
half_rgb = [(int(x*y[0]+0.5))*yy[0]*csc/(x*csc) for x in fullrange]
quar_rgb = [(int(x*y[1]+0.5))*yy[1]*csc/(x*csc) for x in fullrange]

plt.figure(1)
plt.xlabel("raw pixle yuv value")
plt.ylabel("calc pixel rgb value error")
plt.title("close to 1 is better")
plt_vig, = plt.plot(range(16, 218), vig_rgb, 'bo')
plt_half, = plt.plot(range(16, 218), half_rgb, 'ro')
plt_quar, = plt.plot(range(16, 218), quar_rgb, 'go')
plt.legend([plt_vig, plt_half, plt_quar], ('Vig_CSC', 'half_YCoCg', 'quar_YCoCg'), 'best', numpoints=1)
plt.savefig("fullrange.jpg")
vig_rgb_lowfq = [int(x*csc+0.5)/(x*csc) for x in lowfq]
half_rgb_lowfq = [(int(x*y[0]+0.5))*yy[0]*csc/(x*csc) for x in lowfq]
quar_rgb_lowfq = [(int(x*y[1]+0.5))*yy[1]*csc/(x*csc) for x in lowfq]

vig_rgb_highfq = [int(x*csc+0.5)/(x*csc) for x in highfq]
half_rgb_highfq = [(int(x*y[0]+0.5))*yy[0]*csc/(x*csc) for x in highfq]
quar_rgb_highfq = [(int(x*y[1]+0.5))*yy[1]*csc/(x*csc) for x in highfq]

plt.figure(2)
plt.subplot(211)
plt.ylabel("calc pixel rgb value error")
plt.title("close to 1 is better")
plt_vig, = plt.plot(lowfq, vig_rgb_lowfq, 'bo')
plt_half, = plt.plot(lowfq, half_rgb_lowfq, 'ro')
plt_quar, = plt.plot(lowfq, quar_rgb_lowfq, 'go')
plt.legend([plt_vig, plt_half, plt_quar], ('Vig_CSC', 'half_YCoCg', 'quar_YCoCg'), 'best', numpoints=1)
plt.subplot(212)
plt.xlabel("raw pixle yuv value")
plt.ylabel("calc pixel rgb value error")
plt_vig, = plt.plot(highfq, vig_rgb_highfq, 'bo')
plt_half, = plt.plot(highfq, half_rgb_highfq, 'ro')
plt_quar, = plt.plot(highfq, quar_rgb_highfq, 'go')
#plt.legend([plt_vig, plt_half, plt_quar], ('Vig_CSC', 'half_YCoCg', 'quar_YCoCg'), 'best', numpoints=1)
plt.savefig("fq.jpg")
plt.show()
