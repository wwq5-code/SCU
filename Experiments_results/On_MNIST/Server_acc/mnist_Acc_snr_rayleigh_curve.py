

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['5', '10', '15', '20', '25']
unl_org = [97.58, 97.64, 97.92, 97.50, 97.72]

unl_hess_r = [00.00, 96.06, 95.56, 95.83, 96.61]
unl_vbu = [00.00, 00.00, 93.83, 90.22, 90.11]

unl_ss_w = [98.03, 97.89, 98.03, 97.94, 98.03]

unl_hess_r_back = [0.14,  0.94, 1.03, 0.75, 1.61]
unl_vbu_back = [8.19,     3.69, 5.22, 8.17, 3.19]

unl_ss_w_back = [8.92, 7.47, 7.25, 7.19, 8.50]
#unl_ss_wo = [94.71, 93.83, 94.78, 94.07, 93.87]



plt.figure()
l_w=5
m_s=15
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_w, color='g',  marker='*',  label='SCU',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu, color='orange',  marker='x',  label='VBU',linewidth=l_w,  markersize=m_s)

plt.plot(x, unl_hess_r, color='r',  marker='p',  label='HBU',linewidth=l_w, markersize=m_s)


plt.plot(x, unl_ss_w_back, color='g',  marker='*', linestyle=(0, (1, 2, 4)), label='SCU (bac.)',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu_back, color='orange',  marker='x', linestyle=(0, (1, 2, 4)),  label='VBU (bac.)',linewidth=l_w,  markersize=m_s)

plt.plot(x, unl_hess_r_back, color='r',  marker='p', linestyle=(0, (1, 2, 4)), label='HBU (bac.)',linewidth=l_w, markersize=m_s)


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Acc. and backdoor acc. (%)', fontsize=20)
my_y_ticks = np.arange(0, 101, 20)
plt.yticks(my_y_ticks, fontsize=20)
plt.xlabel('$\it{SNR}$', fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')
plt.legend(loc='best', fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_acc_snr_rayleigh_curve.png', dpi=200)
plt.show()