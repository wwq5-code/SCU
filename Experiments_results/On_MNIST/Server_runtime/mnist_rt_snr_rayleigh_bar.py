import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['20%', '40%', '60%', '80%', '100%']
unl_fr = [275, 275, 275, 275, 275]


unl_hess_r = [1.375*10+11*1.375/6000*16, 1.375*10+7*1.375/6000*16 , 1.375*10+5*1.375/6000*16, 1.375*10+5*1.375/6000*16, 1.375*10+6*1.375/6000*16]
unl_vbr = [148*1.375/6000*16, 100*1.375/6000*16, 26*1.375/6000*16, 26*1.375/6000*16, 73*1.375/6000*16]


unl_vib = [128*1.375/6000*16, 121*1.375/6000*16, 736*1.375/6000*16, 121*1.375/6000*16, 119*1.375/6000*16]
unl_self_r = [397*2*1.375/6000*16, 304*2*1.375/6000*16, 320*2*1.375/6000*16, 331*2*1.375/6000*16, 259*2*1.375/6000*16]



x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)


plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_vbr, width=0.168, label='VBU', color='orange', hatch='\\')
plt.bar(x - width / 8 - width / 16,   unl_vbr, width=0.168, label='VBU', color='orange', hatch='\\') #unl_vib, width=0.168, label='MCFU$_{w}$', color='palegreen', hatch='/')
plt.bar(x + width / 8, unl_self_r, width=0.168, label='SCU', color='g', hatch='x')
plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBU', color='tomato', hatch='-')


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

# my_y_ticks = np.arange(0, 3.1, 0.5)
my_y_ticks = np.arange(0, 16.1, 3)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.legend(loc='upper left', fontsize=20)
plt.xlabel('$\it{SNR}$', fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_rt_snr_rayleigh_bar.png', dpi=200)
plt.show()
