# different length of Q_exe M
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict


# def smooth_curve(data, smoothing_factor=0.9):
#     """
#     平滑拟合函数，将数据在 TensorBoard 中进行平滑显示
#
#     参数:
#         - data: 要平滑拟合的原始数据，通常是一个列表或 NumPy 数组
#         - smoothing_factor: 平滑系数，取值范围为 [0, 1]，值越大平滑效果越明显
#
#     返回:
#         平滑拟合后的数据
#     """
#     smoothed_data = []
#     last_smoothed_value = data[0]  # 使用原始数据的首个值初始化平滑后的数据
#
#     for value in data:
#         smoothed_value = last_smoothed_value * smoothing_factor + value * (1 - smoothing_factor)
#         smoothed_data.append(smoothed_value)
#         last_smoothed_value = smoothed_value
#
#     return smoothed_data


def _tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = smooth + 1
    x = values[0]
    res = [x]
    for i in range(1, len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


fig, ax = plt.subplots()

# proposed scheme: Q_exe length = 5
proposed_scheme = pd.read_csv('./2023-09-28 19-36-59/Proposed_Train_Rewards.csv')
proposed_scheme_rewards = proposed_scheme[['Value']]
proposed_scheme_rewards_list = proposed_scheme_rewards.values.tolist()
proposed_scheme_smooth_rewards = _tensorboard_smoothing(np.round(proposed_scheme_rewards_list, 2), 0.95)

# proposed scheme: Q_exe length = 2
Q2_scheme = pd.read_csv('./2023-10-29 10-33-15-Q_exe-2/Train_Rewards.csv')
Q2_scheme_rewards = Q2_scheme[['Value']]
Q2_scheme_rewards_list = Q2_scheme_rewards.values.tolist()
Q2_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Q2_scheme_rewards_list, 2), 0.95)

# Q_exe length = 4
Q4_scheme = pd.read_csv('./2023-10-28 22-10-00-Q_exe-4/Train_Rewards.csv')
Q4_scheme_rewards = Q4_scheme[['Value']]
Q4_scheme_rewards_list = Q4_scheme_rewards.values.tolist()
Q4_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Q4_scheme_rewards_list, 2), 0.95)

# Q_exe length = 6
Q6_scheme = pd.read_csv('./2023-10-29 12-46-58-Q_exe-6/Train_Rewards.csv')
Q6_scheme_rewards = Q6_scheme[['Value']]
Q6_scheme_rewards_list = Q6_scheme_rewards.values.tolist()
Q6_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Q6_scheme_rewards_list, 2), 0.95)

# A scheme: Q_exe length = 7
A_scheme = pd.read_csv('./2023-10-04 14-24-39-Q_exe-7/Train_Rewards.csv')
A_scheme_rewards = A_scheme[['Value']]
A_scheme_rewards_list = A_scheme_rewards.values.tolist()
A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.95)

# Q_exe length = 8
Q8_scheme = pd.read_csv('./2023-10-29 16-36-56-Q_exe-8/Train_Rewards.csv')
Q8_scheme_rewards = Q8_scheme[['Value']]
Q8_scheme_rewards_list = Q8_scheme_rewards.values.tolist()
Q8_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Q8_scheme_rewards_list, 2), 0.95)

# B scheme: Q_exe length = 10
B_scheme = pd.read_csv('./2023-10-04 09-21-53-Q_exe-10/Train_Rewards.csv')
B_scheme_rewards = B_scheme[['Value']]
B_scheme_rewards_list = B_scheme_rewards.values.tolist()
B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.95)

# 绘图  B 优于 A 优于 D 优于 C
x = np.arange(0, len(A_scheme_rewards), 1)

ax.plot(x, Q2_scheme_smooth_rewards, label="M=2", color="#000000", zorder=2)

ax.plot(x, Q4_scheme_smooth_rewards, label="M=4", color="#548B54", zorder=2)

ax.plot(x, proposed_scheme_smooth_rewards, label="M=5", color="#FF00FF", zorder=1)

ax.plot(x, Q6_scheme_smooth_rewards, label="M=6", color="#FF0000", zorder=2)

# ax.plot(x, A_scheme_smooth_rewards, label="M=7", color="#FF7043", zorder=6)

ax.plot(x, Q8_scheme_smooth_rewards, label="M=8", color="#0000FF", zorder=2)

# ax.plot(x, B_scheme_smooth_rewards, label="M=10", color="#0000FF", zorder=5)


default_ticks = np.linspace(0, 1000, 7)
new_ticks = np.linspace(0, 6, 7)
x_ticklabels = [str(int(tick)) for tick in new_ticks]
plt.xticks(default_ticks, x_ticklabels)

plt.xlabel('Episode (1e4)', fontsize=12)
plt.ylabel('Rewards', fontsize=12)
plt.grid()
plt.legend(loc='lower right', prop={'size': 12})  # 显示标签

plt.show()
