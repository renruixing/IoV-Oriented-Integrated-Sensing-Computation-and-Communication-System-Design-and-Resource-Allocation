# average service time delay comparison: proposed scheme VS DDPG scheme VS average allocation VS random allocation
import matplotlib.pyplot as plt
import numpy as np

proposed_scheme = [98.75, 102.58, 132.51, 181.41, 232.94, 255.22, 270.45]
DDPG_scheme = [122.64, 132.61, 186.36, 233.03, 258.00, 271.66, 281.49]
average_allocation_scheme = [134.52, 163.64, 237.09, 266.88, 278.13, 282.05, 288.14]
random_allocation_scheme = [140.56, 175.62, 244.41, 269.34, 279.18, 283.80, 289.51]

x = np.arange(10, 80, 10)
fig, ax = plt.subplots()

ax.plot(x, random_allocation_scheme, 'ms-', linewidth=1, label='Random allocation scheme')
ax.plot(x, average_allocation_scheme, 'bx-', linewidth=1, label='Average allocation scheme')
ax.plot(x, DDPG_scheme, 'g*-', linewidth=1, label='DDPG scheme')
ax.plot(x, proposed_scheme, 'yd-', linewidth=1, label='Proposed scheme')

plt.grid()
plt.legend(loc='lower right', prop={'size': 11})  # 显示标签
plt.xlabel('Number of vehicles', fontsize=12)  # 设置X轴名称
plt.ylabel('Average service time (ms)', fontsize=12)  # 设置Y轴名称
# plt.savefig('Fig5', dpi=1200, bbox_inches='tight')
plt.show()

