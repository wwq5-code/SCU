

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
unl_org = [3.2772, 3.0893, 2.8695, 2.7420, 2.7011]

unl_hess_r = [41.8482, 69.5645, 264.1649, 7.7290, 7.4033]
unl_vbu = [47.4349, 373.5517, 52.0520, 144.0865, 4.9901]

unl_ss_w = [3.212, 3.1021, 2.8842, 2.7372, 2.7072]
#unl_ss_wo = [6.04, 21.44, 31.33, 42.16, 45.34]

unl_hess_r_back = [48.4942, 75.6450, 270.7839, 12.0696, 11.8375]
unl_vbu_back = [53.2619, 377.5891, 56.5367, 149.3794, 8.5334]

unl_ss_w_back = [6.3768, 6.2129, 6.1776, 5.9444, 6.0008]

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

plt.text(1, 7.7983, f'{6.37}', fontsize=20, ha='center',va='bottom')

plt.text(5, 7.7983, f'{5.94}', fontsize=20, ha='center',va='bottom')


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MSE' ,fontsize=20)
my_y_ticks = np.arange(0, 401, 80)
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
plt.savefig('mnist_mse_clean_snr_rician_curve.pdf', format='pdf', dpi=200)
plt.show()