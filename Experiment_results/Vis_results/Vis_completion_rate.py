# average service time delay comparison: proposed scheme VS DDPG scheme VS average allocation VS random allocation
import matplotlib.pyplot as plt
import numpy as np

proposed_scheme = [0.97, 0.93, 0.88, 0.83, 0.71, 0.53, 0.41]
DDPG_scheme = [0.91, 0.85, 0.76, 0.64, 0.51, 0.39, 0.31]
average_allocation_scheme = [0.93, 0.82, 0.67, 0.35, 0.22, 0.15, 0.11]
random_allocation_scheme = [0.81, 0.72, 0.60, 0.32, 0.20, 0.14, 0.10]

x = np.arange(10, 80, 10)
fig, ax = plt.subplots()

ax.plot(x, proposed_scheme, 'yd-', linewidth=1, label='Proposed scheme')
ax.plot(x, DDPG_scheme, 'g*-', linewidth=1, label='DDPG scheme')
ax.plot(x, average_allocation_scheme, 'bx-', linewidth=1, label='Average allocation scheme')
ax.plot(x, random_allocation_scheme, 'ms-', linewidth=1, label='Random allocation scheme')

plt.grid()
plt.legend(loc='lower left', prop={'size': 10})  # 显示标签
plt.xlabel('Number of vehicles', fontsize=12)  # 设置X轴名称
plt.ylabel('Average task completion rate', fontsize=12)  # 设置Y轴名称
# plt.savefig('Fig5', dpi=1200, bbox_inches='tight')
plt.show()

