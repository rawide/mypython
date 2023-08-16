import numpy as np
import matplotlib.pyplot as plt

def show_text(ax, mat, txt=''):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, txt+'%s%s'%(i,j), ha='center', va='center', color='white')

def draw_matmul(a, b, c, src1='x', src2='w_q', dst='Q'):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap('viridis')
    ax[0].imshow(a, cmap='viridis')
    ax[0].set_title(src1)
    show_text(ax[0], a, src1)

    ax[1].imshow(b, cmap=cmap)
    ax[1].set_title(src2)
    show_text(ax[1], b, src2)

    ax[2].imshow(c, cmap=cmap)
    ax[2].set_title(dst)
    show_text(ax[2], c, dst)


x = np.array([[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]])
w_q = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
Q = np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34]])
# draw_matmul(x, w_q, Q, 'x', 'w_q', 'Q')

x1 = np.array([[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
w_q1 = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])
Q1 = np.array([[11,0,0,0],[0,0,0,0],[0,0,0,0]])
# draw_matmul(x1, w_q1, Q1, 'x', 'w_q', 'Q')

x1 = np.array([[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
w_q1 = np.array([[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]])
Q1 = np.array([[0,11,0,0],[0,0,0,0],[0,0,0,0]])
# draw_matmul(x1, w_q1, Q1, 'x', 'w_q', 'Q')

Q_kt = np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0]])
Kt = np.array([[1,2,0,0],[1,2,0,0],[1,2,0,0]])
attn = np.array([[11,11,0],[0,0,0],[0,0,0]])
draw_matmul(Q_kt, Kt, attn, 'Q', 'Kt', 'attn')
plt.show()
print("minimal memory size is num_mac + num_mac + 1 + 1 [input, weight, bias, output]")
'''
memory grounp should be integer times of num_mac, consider the DDR->SRAM transfer speed, 
a memory group queue need to be built for mac using, assuming:
size_memgrp = 2*num_mac+2: size of one memory group, for mac using 
depth_memgrp = speed_ddr_sram/(freq_Mac*size_memgrp): depth of memory group queue
size_sram = (size_memgrp * cnt_memgrp)*pingpong_cnt
sram = f(mac)
'''
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# cmap = plt.get_cmap('viridis')
# ax[0].imshow(x, cmap='viridis')
# ax[0].set_title("input [3,8]")
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         ax[0].text(j, i, 'x%s%s'%(i,j), ha='center', va='center', color='white')

# ax[1].imshow(w_q, cmap=cmap)
# ax[1].set_title("head1: w_q [8,4]")
# for i in range(w_q.shape[0]):
#     for j in range(w_q.shape[1]):
#         ax[1].text(j, i, 'w_q%s%s'%(i,j), ha='center', va='center', color='white')

# ax[2].imshow(Q, cmap=cmap)
# ax[2].set_title("Q1:[3,4]")
# for i in range(Q.shape[0]):
#     for j in range(Q.shape[1]):
#         ax[2].text(j, i, 'Q%s%s'%(i,j), ha='center', va='center', color='white')



