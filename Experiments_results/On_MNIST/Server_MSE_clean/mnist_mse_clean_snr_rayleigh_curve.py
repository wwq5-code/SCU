

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
unl_org = [2.7541, 2.7412, 2.6454, 2.6283, 2.6845]

unl_hess_r = [55.8346, 9.5056, 5.3834, 4.7754, 8.7987]
unl_vbu = [132.4361, 76.1527, 7.6597, 10.2683, 50.5719]

unl_ss_w = [2.7530, 2.7311, 2.6309, 2.6076, 2.6847]
#unl_ss_wo = [6.04, 21.44, 31.33, 42.16, 45.34]


unl_hess_r_back = [64.8582, 14.4627, 9.3798, 8.8702, 13.8774]
unl_vbu_back = [138.67, 79.6193, 11.1670, 13.7284, 54.5881]

unl_ss_w_back = [6.1200, 6.0555, 5.9701, 5.9500, 6.0492]

plt.style.use('seaborn')
plt.figure()

l_w = 5
m_s = 15
marker_s = 3
markevery = 1

# plt.figure(figsize=(8, 5.3))
# plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)


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

plt.plot(x, unl_hess_r_back, linestyle=(0, (1, 2, 4)), color='#E07B54', marker='D', fillstyle='full',
         markevery=markevery,
         label='HBU (bac.)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.text(1, 7.7983, f'{6.12}', fontsize=20, ha='center',va='bottom')

plt.text(5, 7.7983, f'{6.05}', fontsize=20, ha='center',va='bottom')


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MSE' ,fontsize=20)
my_y_ticks = np.arange(0, 161, 40)
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
plt.savefig('mnist_mse_clean_snr_rayleigh_curve.pdf', format='pdf', dpi=200)
plt.show()