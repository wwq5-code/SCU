

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[1, 2, 3, 4, 5]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['5', '10', '15', '20', '25']
unl_org = [97.17, 97.56, 97.64, 97.58, 97.42]

unl_hess_r = [0.04, 93.04, 61.69, 90.86, 64.72]
unl_vbu = [68.19,   88.19, 94.19, 68.22, 83.78]

unl_ss_w = [97.56, 97.92, 97.86, 97.72, 97.86]

unl_hess_r_back = [0.06,  0.67, 5.64, 1.08, 0.56]
unl_vbu_back = [2.11,     2.83, 6.61, 5.83, 4.92]

unl_ss_w_back = [8.97, 8.81, 7.53, 9.47, 10.06]
#unl_ss_wo = [94.71, 93.83, 94.78, 94.07, 93.87]


plt.style.use('seaborn')
plt.figure()

l_w=5
m_s=15
marker_s = 3
markevery=1

#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)






plt.plot(x, unl_ss_w, linestyle='-', color='#9BC985', marker='o', fillstyle='full', markevery=markevery,
         label='SCU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



plt.plot(x, unl_vbu, linestyle='-', color='#797BB7', marker='s', fillstyle='full', markevery=markevery,
         label='VBU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, unl_hess_r, linestyle='-', color='#E07B54', marker='D', fillstyle='full', markevery=markevery,
         label='HBU', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



plt.plot(x, unl_ss_w_back, linestyle=(0, (1, 2, 4)), color='#9BC985', marker='o', fillstyle='full', markevery=markevery,
         label='SCU (bac.)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



plt.plot(x, unl_vbu_back, linestyle=(0, (1, 2, 4)), color='#797BB7', marker='s', fillstyle='full', markevery=markevery,
         label='VBU (bac.)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, unl_hess_r_back, linestyle=(0, (1, 2, 4)), color='#E07B54', marker='D', fillstyle='full', markevery=markevery,
         label='HBU (bac.)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
# leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Acc. and backdoor acc. (%)', fontsize=24)
my_y_ticks = np.arange(0, 101, 20)
plt.yticks(my_y_ticks, fontsize=20)
plt.xlabel('$\it{SNR}$', fontsize=24)

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
plt.savefig('mnist_acc_snr_awgn_curve.pdf', format='pdf', dpi=200)
plt.show()