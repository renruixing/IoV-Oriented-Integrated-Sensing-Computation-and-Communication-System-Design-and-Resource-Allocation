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


# proposed scheme: ②1次step训练1次critic，2次step训练1次actor网络（有延迟更新策略），target network都是100个step更新一次 2023-09-28 23-57-30
proposed_scheme = pd.read_csv('./2023-09-28 19-36-59/Proposed_Train_Rewards.csv')
proposed_scheme_rewards = proposed_scheme[['Value']]
proposed_scheme_rewards_list = proposed_scheme_rewards.values.tolist()
proposed_scheme_smooth_rewards = _tensorboard_smoothing(np.round(proposed_scheme_rewards_list, 2), 0.95)

# B_1 scheme: Based proposed scheme 前期无探索率
B_1_scheme = pd.read_csv('./2023-09-30 09-41-47/Train_Rewards.csv')
B_1_scheme_rewards = B_1_scheme[['Value']]
B_1_scheme_rewards_list = B_1_scheme_rewards.values.tolist()
B_1_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_1_scheme_rewards_list, 2), 0.95)


x = np.arange(0, len(proposed_scheme_rewards), 1)
ax.plot(x, proposed_scheme_smooth_rewards, label="proposed scheme", color="#0000FF", zorder=5)  # actor有延迟更新策略，target网络更新率100步一次
ax.plot(x, B_1_scheme_smooth_rewards, label="no exploration rates scheme", color="#000000", zorder=1)  # Based on B scheme 前期无探索
default_ticks = np.linspace(0, 1000, 6)
new_ticks = np.linspace(0, 6, 6)
x_ticklabels = [str(int(tick)) for tick in new_ticks]
plt.xticks(default_ticks, x_ticklabels)

plt.xlabel('episode (1e4)', fontsize=12)
plt.ylabel('rewards', fontsize=12)
plt.legend()  # 显示标签

plt.show()













