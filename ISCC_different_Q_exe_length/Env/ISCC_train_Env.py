import random

import numpy as np
import math as m


def softmax(x):
    # 计算e的x次方
    exp_vals = np.exp(x)
    # 计算每个类别的概率分布
    softmax_vals = exp_vals / np.sum(exp_vals)
    return softmax_vals


def transmission_rate(num, Tx_power_dBm, B_ratio, B_resource):
    """
    :param num:
    :param Tx_power_dBm: dBm
    :param B_ratio:
    :param B_resource: MHz
    noise: -174 dBm/Hz
    :return:
    """
    B_resource_Hz = B_resource * 1e6
    Tx_power_W = 10 ** (Tx_power_dBm / 10) * 1e-3
    noise_power = 10 ** (-174 / 10) * 1e-3 * B_resource_Hz
    sum_rate = []
    for i in range(num):
        rate_bps = B_resource_Hz * B_ratio[i] * (m.log2(1 + Tx_power_W / noise_power))
        rate_Mbpms = rate_bps / 1e3 / 1e6  # Mbits/ms
        sum_rate.append(rate_Mbpms * 10)  # todo
    return sum_rate


class env:
    def __init__(self, gamma):
        self.tao = 10  # length of time-slot tao: 10ms
        # self.T = num_timeslots  # number of time-slots
        self.R_u = 10  # uplink bandwidth of RSU: 10MHz
        self.R_d = 5  # downlink bandwidth of RSU: 5MHz
        self.R_c = 20  # computing resources of MEC server: 2.5 × 4 Core GHz
        self.P_i = 30  # Tx power of vehicles: 30 dBm
        self.P_RSU = 27  # Tx power of RSU: 27 dBm
        self.M = 8  # Q_exe length M: 5
        self.f_v = 1e9  # computing resources of vehicle: 2GHz
        self.done = 0  # 所有任务完成则为1
        self.gamma = gamma

        # init_task
        self.r_sep = 0.2  # Compression coefficient of vehicle side sensing data before uploading
        # self.task = Task_generation(num_task=self.N, num_timeslots=self.T, compression_coefficient=self.r_sep)

        # init Q_exe state space
        self.time_state = [0] * self.M
        self.up_state = [0] * self.M
        self.computing_state = [0] * self.M
        self.down_state = [0] * self.M

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.M

        self.num_tasks = 0
        self.Task_done = [0] * self.M

        # init phase
        self.up_phase = [0] * self.M
        self.computing_phase = [0] * self.M
        self.down_phase = [0] * self.M

        # init task size
        self.upload_data = [0] * self.M
        self.computation_data = [0] * self.M
        self.download_data = [0] * self.M
        self.delay_constraint = [0] * self.M

    def reset_state(self):
        self.M = 8  # Q_exe length M: 5
        self.Q_exe = [0] * self.M
        self.num_tasks = 0
        self.Task_done = [0] * self.M

        # init phase
        self.up_phase = [0] * self.M
        self.computing_phase = [0] * self.M
        self.down_phase = [0] * self.M

        # init task size
        self.upload_data = [0] * self.M
        self.computation_data = [0] * self.M
        self.download_data = [0] * self.M
        self.delay_constraint = [0] * self.M

        # init Q_exe state space
        self.time_state = [0] * self.M
        self.up_state = [0] * self.M
        self.computing_state = [0] * self.M
        self.down_state = [0] * self.M

        for i in range(self.M):
            a = random.random()
            if a > self.gamma:
                self.Q_exe[i] = 0
            else:
                self.Q_exe[i] = 1
                self.num_tasks += 1
        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] == 1:
                self.upload_data[i] = random.randint(10, 30)
                self.computation_data[i] = random.randint(100, 200)
                self.download_data[i] = random.randint(5, 15)
                self.delay_constraint[i] = random.randint(100, 500)
                b = random.random()
                if 0 <= b <= 0.3:
                    # 上传阶段
                    self.up_phase[i] = 1
                    # init state
                    up_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.up_state[i] = up_data_rest_ratio * self.upload_data[i]
                    self.computing_state[i] = 1 * self.computation_data[i]
                    self.down_state[i] = 1 * self.download_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.9
                if 0.3 < b < 0.7:
                    # 计算阶段
                    self.computing_phase[i] = 1
                    self.up_state[i] = 0
                    computing_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.computing_state[i] = computing_data_rest_ratio * self.computation_data[i]
                    self.down_state[i] = 1 * self.download_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.6
                if 0.7 <= b <= 1:
                    # 下载阶段
                    self.down_phase[i] = 1
                    self.up_state[i] = 0
                    self.computing_state[i] = 0
                    down_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.down_state[i] = down_data_rest_ratio * self.download_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.3
        state = self.up_state + self.computing_state + self.down_state + self.time_state

        return state

    def step(self, action):
        self.P_i = 30  # Tx power of vehicles: 30 dBm
        self.P_RSU = 27  # Tx power of RSU: 27 dBm

        # up_B_ratio = softmax(np.array(action[:5])).tolist()
        # computing_resource_ratio = softmax(np.array(action[5:10])).tolist()
        # down_B_ratio = softmax(np.array(action[-5:])).tolist()

        up_B_ratio = action[:8]
        computing_resource_ratio = action[8:16]
        down_B_ratio = action[-8:]

        reward = 0
        up_rate = transmission_rate(len(action[:8]), self.P_i, up_B_ratio, self.R_u)
        computing_rate = []
        for i in range(len(computing_resource_ratio)):
            computing_rate.append(computing_resource_ratio[i] * self.R_c * 1e3 / 1e3)  # self.R_c * 1e3 => MHz,即Mcycles/s, 除以1000 => Mcycles/ms
        down_rate = transmission_rate(len(action[:8]), self.P_RSU, down_B_ratio, self.R_d)

        # print('当前上行链路速率情况：Mbits/ms', up_rate)
        # print('当前计算速率分配情况：Mcycles/ms', computing_rate)
        # print('当前下行链路速率情况：Mbits/ms', down_rate)
        # print('当前时隙任务运行 前 的任务运行队列:{}'.format(self.Q_exe))
        # print('当前时隙任务运行 前 任务运行队列的剩余 上传 任务大小:{}'.format(self.up_state))
        # print('当前时隙任务运行 前 任务运行队列的剩余 计算 任务大小:{}'.format(self.computing_state))
        # print('当前时隙任务运行 前 任务运行队列的剩余 下载 任务大小:{}'.format(self.down_state))
        # print('当前时隙任务运行 前 任务运行队列的剩余可用执行时间:{}'.format(self.time_state))
        # print('**开**始**执*行**')
        reward_list = []
        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] != 0:
                self.time_state[i] = self.time_state[i] - self.tao
                # up_rate: Mbits/ms
                self.up_state[i] -= self.up_phase[i] * self.tao * up_rate[i]
                # computing_rate: Mcycles/ms
                self.computing_state[i] -= self.computing_phase[i] * self.tao * computing_rate[i]
                self.down_state[i] -= self.down_phase[i] * self.tao * down_rate[i]

                # 如果任务i的时间状态小于等于0 并且 下载状态大于0，任务直接未完成
                if self.time_state[i] <= 0 and self.down_state[i] > 0:
                    # print(f'第{i}个任务 未 完成，已将Q_exe中移除-------------------------')
                    self.Task_done[i] = -1
                    self.Q_exe[i] = 0
                    self.time_state[i] = 0
                    self.up_state[i] = 0
                    self.computing_state[i] = 0
                    self.down_state[i] = 0
                else:  # todo 这我觉得还可以改进 23.9.9
                    if self.up_state[i] <= 0:
                        self.up_state[i] = 0

                        self.up_phase[i] = 0
                        self.computing_phase[i] = 1

                    if self.computing_state[i] <= 0:
                        self.computing_state[i] = 0

                        self.computing_phase[i] = 0
                        self.down_phase[i] = 1

                    if self.down_state[i] <= 0:
                        self.down_state[i] = 0
                        self.time_state[i] = 0

                        self.down_phase[i] = 0
                        self.Task_done[i] = 1
                        self.Q_exe[i] = 0
                        # print(f'第{i}个任务 已 完成，已将Q_exe中移除++++++++++++++++++++++++++')
            # todo 奖励这里还是有点问题
            reward += (self.Task_done[i] * 5 - (2 - self.time_state[i] / 500) * (self.up_state[i] / 30 + self.computing_state[i] / 200 + self.down_state[i] / 15) / 3)

        average_reward = reward / self.num_tasks
        # print('当前时隙任务运行完之 后 的任务运行队列:{}'.format(self.Q_exe))
        # print('当前时隙任务运行完之 后 任务运行队列的剩余 上传 任务大小:{}'.format(self.up_state))
        # print('当前时隙任务运行完之 后 任务运行队列的剩余 计算 任务大小:{}'.format(self.computing_state))
        # print('当前时隙任务运行完之 后 任务运行队列的剩余 下载 任务大小:{}'.format(self.down_state))
        # print('当前时隙任务运行完之 后 任务运行队列的剩余可用执行时间:{}'.format(self.time_state))
        # print('当前时隙任务完成情况：{}'.format(self.Task_done))
        new_state = self.up_state + self.computing_state + self.down_state + self.time_state
        count_complete = 0
        for i in range(len(self.Task_done)):
            if self.Task_done[i] == 1 or self.Task_done[i] == -1:
                count_complete += 1
        if count_complete == self.num_tasks:
            self.done = True
        else:
            self.done = False
        return new_state, average_reward, self.done
