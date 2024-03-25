

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['2%', '4%', '6%', '8%', '10%' ]
unl_org = [96.08, 97.00, 97.72, 97.40, 98.12]

unl_hess_r = [3.42,  4.46, 2.06, 3.10, 3.27]
unl_vbu = [3.67,     4.92, 2.11, 4.35, 4.40]

unl_ss_w = [7.17,  8.75, 7.28,  7.12, 8.55]
#unl_ss_wo = [1.11, 0.66, 0.75, 0.6, 0.03]



plt.style.use('seaborn')
# plt.figure()
plt.figure(figsize=(6.5, 5.3))
l_w=5
m_s=15

marker_s = 3
markevery=1

#plt.figure(figsize=(8, 5.3))
# plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
# plt.plot(x, unl_ss_w, color='g',  marker='*',  label='SCU',linewidth=l_w, markersize=m_s)
# #plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)
#
# plt.plot(x, unl_vbu, color='orange',  marker='x',  label='VBU',linewidth=l_w,  markersize=m_s)
#
# plt.plot(x, unl_hess_r, color='deepskyblue',  marker='p',  label='HBU',linewidth=l_w, markersize=m_s)



plt.plot(x, unl_ss_w, linestyle='-', color='#9BC985', marker='o', fillstyle='full', markevery=markevery,
         label='SCU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, unl_hess_r, linestyle='-', color='#ECC97F', marker='D', fillstyle='full', markevery=markevery,
         label='SCU w/o CC', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu, linestyle='--', color='#797BB7',  marker='s', fillstyle='full', markevery=markevery,
         label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Backdoor Accuracy (%)' ,fontsize=24)
my_y_ticks = np.arange(0, 11, 2)
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
plt.savefig('mnist_backacc_er_curve_ablation.pdf', format='pdf', dpi=200)
plt.show()