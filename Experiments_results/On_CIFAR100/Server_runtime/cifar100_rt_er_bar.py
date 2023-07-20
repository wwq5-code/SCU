import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['2%', '4%', '6%', '8%', '10%']
unl_fr = [2120 , 2120, 2120 , 2120 , 2120]

unl_hess_r = [106 + 3*106/50000*16, 106 + 3*106/50000*16, 106 + 2*106/50000*16, 106 + 2*106/50000*16, 106 + 2*106/50000*16]
unl_vbr = [7*1.6/21 /5*21, 6*1.6/21 /5*21, 6*1.6/21 /5*21, 6*1.6/21 /5*21, 4*1.6/21 /5*21]


unl_vib = [10*106/50000*16, 11*106/50000*16, 7*106/50000*16, 6*106/50000*16, 7*106/50000*16]
unl_self_r = [13*2.23/23 /5*23, 14*2.23/23/5*23, 7*2.23/23/5*23, 6*2.23/23/5*23, 9*2.23/23/5*23]


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


# plt.bar(x - width / 2.5 ,  unl_vbr, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 125, 30)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

plt.legend(loc='upper left', fontsize=20)
plt.xlabel('$\it{EDR}$' ,fontsize=20)
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
plt.savefig('cifar100_rt_er_bar.png', dpi=200)
plt.show()
