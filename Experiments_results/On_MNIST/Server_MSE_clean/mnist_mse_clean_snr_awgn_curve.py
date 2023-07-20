

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['5', '10', '15', '20', '25']
#unl_org_0=[28.59, 28.59, 28.59, 28.59, 28.59]
unl_org = [2.7909, 2.7087, 2.7082, 2.6308, 2.6083]

unl_hess_r = [74.9007, 9.6105, 16.6510, 14.4512, 29.0367]
unl_vbu = [71.9451, 45.9706, 8.4615, 25.311, 20.1302]

unl_ss_w = [2.7983, 2.6868, 2.7025, 2.6176, 2.5964]
#unl_ss_wo = [6.04, 21.44, 31.33, 42.16, 45.34]

unl_hess_r_back = [83.3344, 14.9082, 22.6269, 20.3483, 35.0522]
unl_vbu_back = [77.4991, 50.5420, 12.081, 29.5716, 24.0316]

unl_ss_w_back = [6.1250, 6.0131, 5.79617, 5.7694, 5.8932]

plt.figure()
l_w=5
m_s=15
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_w, color='g',  marker='*',  label='SCU',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu, color='orange',  marker='x',  label='VBU',linewidth=l_w,  markersize=m_s)

plt.plot(x, unl_hess_r, color='r',  marker='p',  label='HBU',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_ss_w_back, color='g',  marker='*', linestyle=(0, (1, 2, 4)), label='SCU (Er.)',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu_back, color='orange',  marker='x', linestyle=(0, (1, 2, 4)),  label='VBU (Er.)',linewidth=l_w,  markersize=m_s)

plt.plot(x, unl_hess_r_back, color='r',  marker='p', linestyle=(0, (1, 2, 4)), label='HBU (Er.)',linewidth=l_w, markersize=m_s)


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MSE' ,fontsize=20)
my_y_ticks = np.arange(0, 101, 20)
plt.yticks(my_y_ticks, fontsize=20)
plt.xlabel('$\it{SNR}$', fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')
plt.legend(loc='upper right', fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_mse_clean_snr_awgn_curve.png', dpi=200)
plt.show()