

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
unl_org = [41.8592, 39.3931, 39.3814, 38.2626, 39.6953]


unl_hess_r = [3935827.1720, 1056.7093, 54.2166, 65.3965, 1603.0663]
unl_vbu = [48.0690, 63.9892, 58.0831, 55.3035, 74.9349]

unl_ss_clean = [42.4134, 47.5628, 43.3111, 44.6024, 43.4938]
unl_ss_erased = [53.2308, 48.5714, 45.1817, 46.7559, 46.0260]
#unl_ss_wo = [23.54, 19.375, 31.33, 33.92, 37.80]


plt.style.use('seaborn')
plt.figure()
l_w=5
m_s=15

marker_s = 3
markevery=1

#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_ss_clean, linestyle='-', color='#9BC985', marker='o', fillstyle='full', markevery=markevery,
         label='SCU (Clean)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, unl_ss_erased, linestyle=':', color='#2A5522',  marker='^', fillstyle='full', markevery=markevery,
         label='SCU (Erased)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_ss_wo, color='palegreen',  marker='1',  label='MCFU$_{w/o}$',linewidth=l_w, markersize=m_s)

plt.plot(x, unl_vbu, linestyle='--', color='#797BB7',  marker='s', fillstyle='full', markevery=markevery,
         label='VBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



plt.plot(x, unl_hess_r, linestyle='-.', color='#E07B54',  marker='D', fillstyle='full', markevery=markevery,
         label='HBU',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('MSE' ,fontsize=24)
my_y_ticks = np.arange(0 ,201,40)
plt.yticks(my_y_ticks,fontsize=24)
plt.ylim(0, 200)
plt.xlabel('$\it{EDR}$' ,fontsize=24)

plt.xticks(x, labels, fontsize=24)
# plt.title('CIFAR10 IID')
plt.legend(loc='upper left',fontsize=24)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar_mse_clean_er_curve.pdf', format='pdf', dpi=200)
plt.show()