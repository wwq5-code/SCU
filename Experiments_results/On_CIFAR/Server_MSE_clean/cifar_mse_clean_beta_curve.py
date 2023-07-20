

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['0.0001', '0.001', '0.01', '0.1', '1' ]
#unl_org_0= [62.36, 26.76, 17.664, 5.737, 3.64]
unl_org = [39.2377, 39.1694, 39.5015, 39.2438, 39.2235]


unl_hess_r = [54.1759, 54.1745, 54.4611, 53.1500, 53.0745]
unl_vbu = [51.2374, 55.3432, 56.3580, 58.0831, 58.1304]

unl_ss_w = [40.7147, 40.7778, 41.3081, 41.9691, 42.1865]
#unl_ss_wo = [47.961, 31.33, 11.943, 2.402, 0.95]



plt.figure()
l_w=5
m_s=15
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_w, color='g',  marker='*',  label='SCU',linewidth=l_w, markersize=m_s)
#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu, color='orange',  marker='x',  label='VBU',linewidth=l_w,  markersize=m_s)

plt.plot(x, unl_hess_r, color='r',  marker='p',  label='HBU',linewidth=l_w, markersize=m_s)


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Mutual Information' ,fontsize=20)
my_y_ticks = np.arange(0 ,201,40)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('$\\beta$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_mse_clean_beta_curve.png', dpi=200)
plt.show()