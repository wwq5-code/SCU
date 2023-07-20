

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['2%', '4%', '6%', '8%', '10%' ]
#unl_org_0= [24.44, 33.15, 26.5, 31.84, 47.90]
unl_org = [27.7576, 29.0005, 26.8730, 28.8287, 27.9779]


unl_hess_r = [912.1368, 939.8773, 467.5802, 90.1646, 150.6723]
unl_vbu = [78.4020, 71.4398, 55.6163, 51.3816, 51.4810]

unl_ss_clean = [34.986, 36.2846, 33.4547, 37.8271, 31.0512]
unl_ss_erased = [36.7558, 39.3063, 35.4111, 41.5716, 33.0236]

#unl_ss_wo = [23.54, 19.375, 31.33, 33.92, 37.80]



plt.figure()
l_w=5
m_s=15
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_clean, color='palegreen',  marker='*',  label='SCU (Clean)',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_ss_erased, color='g',  marker='1',  label='SCU (Erased)',linewidth=l_w, markersize=m_s)

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
plt.ylabel('MSE' ,fontsize=20)
my_y_ticks = np.arange(0 ,201,40)
plt.yticks(my_y_ticks,fontsize=20)
plt.ylim(0,200)
plt.xlabel('$\it{EDR}$' ,fontsize=20)

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
plt.savefig('cifar100_mse_erased_er_curve.png', dpi=200)
plt.show()