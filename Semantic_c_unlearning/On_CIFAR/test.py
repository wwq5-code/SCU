import numpy as np
import matplotlib.pyplot as plt

# 假设我们有 10 次实验的结果
num_experiments = 10
x = np.linspace(0, 10, num_experiments)
y = np.random.randn(num_experiments)  # 随机生成 10 次实验的结果



# 计算平均值和标准差
y_mean = y.mean(axis=0)
y_std = y.std(axis=0)

y_mean = np.full_like(x, y_mean)
y_std = np.full_like(x, y_std)


print(x)
print(y_mean)
# 画出平均值
plt.plot(x, y_mean, label='Mean')

# 用 fill_between 画出标准差的区间
plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

plt.legend()
plt.show()
