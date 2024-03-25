

import numpy as np
import matplotlib.pyplot as plt

# epsilon = 3
# beta = 1 / epsilon


# snr = 5, beta_u = 0.1 on mnist

x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['2%', '4%', '6%', '8%', '10%' ]
unl_org = [97.83, 97.96, 97.17, 97.71, 97.02]

unl_hess_r = [57.50, 66.50, 70.04, 70.94, 0.85]
unl_vbu = [52.00, 75.67, 68.19, 76.67, 0.76]

unl_ss_w = [98.08, 98.25, 97.56, 97.90, 97.42]
#unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04]


plt.style.use('seaborn')
#plt.figure()
plt.figure(figsize=(6.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1

#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_ss_w, linestyle='-', color='#9BC985', marker='o', fillstyle='full', markevery=markevery,
         label='SCU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

# plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_hess_r, linestyle='-', color='#ECC97F', marker='D', fillstyle='full', markevery=markevery,
         label='SCU w/o CC', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
plt.plot(x, unl_vbu, linestyle='--', color='#797BB7', marker='s', fillstyle='full', markevery=markevery,
         label='VBU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)





#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
# leg = plt.legend(fancybox=True, shadow=True, fontsize=24)
# plt.setp(leg.get_texts(), fontsize=30)  # or use a numerical value

# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(0 ,101,20)
plt.yticks(my_y_ticks,fontsize=24)
plt.xlabel('$\it{EDR}$' ,fontsize=24)

plt.xticks(x, labels, fontsize=24)
# plt.title('CIFAR10 IID')
plt.legend(loc='best',fontsize=24)

plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_acc_er_curve_ablation.pdf', format='pdf', dpi=200)
plt.show()