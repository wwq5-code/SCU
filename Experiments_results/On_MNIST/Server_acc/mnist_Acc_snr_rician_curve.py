

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['5', '10', '15', '20', '25']
unl_org = [95.36, 98.03, 97.86, 97.56, 97.81]

unl_hess_r = [11.00, 0.00, 0.00, 97.00, 97.17]
unl_vbu = [13.94,   0.00, 0.53, 0.00, 97.33]

unl_ss_w = [97.94, 98.08, 98.19, 97.86, 97.97]
#unl_ss_wo = [94.71, 93.83, 94.78, 94.07, 93.87]


unl_hess_r_back = [0.06,  0.08, 0.22, 1.56, 0.67]
unl_vbu_back = [6.11,     9.83, 9.44, 9.58, 5.44]

unl_ss_w_back = [17.72, 9.89, 10.19, 9.33, 7.83]


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
plt.savefig('mnist_acc_snr_rician_curve.png', dpi=200)
plt.show()