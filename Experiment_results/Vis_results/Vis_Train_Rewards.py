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

# A scheme: ①1次step训练1次critic与actor网络（无延迟更新actor策略），target网络都是100个step更新一次 2023-9-28 19-36-59
A_scheme = pd.read_csv('./2023-09-28 23-57-30/Train_Rewards.csv')
A_scheme_rewards = A_scheme[['Value']]
A_scheme_rewards_list = A_scheme_rewards.values.tolist()
A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.95)

# proposed scheme: ②1次step训练1次critic，2次step训练1次actor网络（有延迟更新策略），target network都是100个step更新一次 2023-09-28 23-57-30
proposed_scheme = pd.read_csv('./2023-09-28 19-36-59/Proposed_Train_Rewards.csv')
proposed_scheme_rewards = proposed_scheme[['Value']]
proposed_scheme_rewards_list = proposed_scheme_rewards.values.tolist()
proposed_scheme_smooth_rewards = _tensorboard_smoothing(np.round(proposed_scheme_rewards_list, 2), 0.95)

# B scheme: ④1次step训练1次critic，2次step训练1次actor网络（有延迟更新策略），target network同步更新（相当于无target网络） 2023-09-29 17-16-18
B_scheme = pd.read_csv('./2023-09-29 17-16-18/Train_Rewards.csv')
B_scheme_rewards = B_scheme[['Value']]
B_scheme_rewards_list = B_scheme_rewards.values.tolist()
B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.95)

# C scheme: ③1次step训练1次critic与actor网络（无延迟更新actor策略），target网络同步更新（相当于无target网络）2023-09-29 08-50-28
C_scheme = pd.read_csv('./2023-09-29 08-50-28/Train_Rewards.csv')
C_scheme_rewards = C_scheme[['Value']]
C_scheme_rewards_list = C_scheme_rewards.values.tolist()
C_scheme_smooth_rewards = _tensorboard_smoothing(np.round(C_scheme_rewards_list, 2), 0.95)

# 绘图  B 优于 A 优于 D 优于 C
x = np.arange(0, len(A_scheme_rewards), 1)

ax.plot(x, proposed_scheme_smooth_rewards, label="Proposed scheme", color="#FF00FF", zorder=1)  # actor有延迟更新策略，target网络同步更新

# ax.plot(x, proposed_scheme_rewards, color="#FFE2D9", alpha=0.5, zorder=3)
ax.plot(x, A_scheme_smooth_rewards, label="A scheme", color="#FF7043", zorder=6)

# ax.plot(x, TD3_scheme_rewards, color="#ADD8E6", alpha=0.3, zorder=2)
ax.plot(x, B_scheme_smooth_rewards, label="B scheme", color="#0000FF", zorder=5)

# ax.plot(x, C_scheme_rewards, color="#90EE90", alpha=0.3, zorder=1)
ax.plot(x, C_scheme_smooth_rewards, label="C scheme", color="#00FF00", zorder=4)  # actor无延迟更新策略，target网络同步更新

# ax.plot(x, D_scheme_rewards, color="#E6BEFF", alpha=0.3, zorder=1)

default_ticks = np.linspace(0, 1000, 7)
new_ticks = np.linspace(0, 6, 7)
x_ticklabels = [str(int(tick)) for tick in new_ticks]
plt.xticks(default_ticks, x_ticklabels)

plt.xlabel('Episode (1e4)', fontsize=12)
plt.ylabel('Rewards', fontsize=12)
plt.grid()
plt.legend(loc='lower right', prop={'size': 12})  # 显示标签

plt.show()
