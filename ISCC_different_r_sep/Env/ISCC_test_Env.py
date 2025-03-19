import numpy as np
import math as m


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


class Task:
    def __init__(self, task_id, arrival_time, delay_threshold, vehicle_sensing_data, computation_data, download_data,
                 priority_weight, r_sep):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.delay_threshold = delay_threshold
        self.upload_data = vehicle_sensing_data * r_sep
        self.computation_data = computation_data
        self.download_data = download_data
        self.priority_weight = priority_weight

        self.rest_available_time = self.delay_threshold

        self.upload_phase = 0
        self.computing_phase = 0
        self.download_phase = 0
        self.waiting_phase = 0


def Task_generation(num_task, num_timeslots, compression_coefficient):
    r_sep = compression_coefficient
    task = {}
    arrival_time = np.random.randint(0, num_timeslots * 0.5, num_task)
    delay_threshold = np.random.randint(100, 500, num_task)
    vehicle_sensing_data = np.random.randint(50, 150, num_task)  # 50MB ~ 150MB
    computation_data = np.random.randint(100, 200, num_task)  # MB
    download_data = np.random.randint(5, 15, num_task)  # MB
    priority_weight = np.zeros(1)  # init
    for i in range(0, num_task):
        task[i] = Task(task_id=i + 1, arrival_time=arrival_time[i], delay_threshold=delay_threshold[i],
                       vehicle_sensing_data=vehicle_sensing_data[i], computation_data=computation_data[i],
                       download_data=download_data[i], priority_weight=priority_weight, r_sep=r_sep)
    return task


class env:
    def __init__(self, num_tasks, num_timeslots):
        self.N = num_tasks  # number of vehicles N
        self.tao = 10  # length of time-slot tao: 10ms
        self.T = num_timeslots  # number of time-slots
        self.R_u = 10  # uplink bandwidth of RSU: 10MHz
        self.R_d = 5  # downlink bandwidth of RSU: 5MHz
        self.R_c = 10  # computing resources of MEC server: 2.5 × 4 Core GHz
        self.P_i = 30  # Tx power of vehicles: 30 dBm
        self.P_RSU = 27  # Tx power of RSU: 27 dBm
        self.M = 5  # Q_exe length M: 5
        self.f_v = 1e9  # computing resources of vehicle: 2GHz
        self.Task_done = [0] * self.N
        self.done = False  # 所有任务完成则为1

        # init_task
        self.r_sep = 0.2  # Compression coefficient of vehicle side sensing data before uploading
        self.task = Task_generation(num_task=self.N, num_timeslots=self.T, compression_coefficient=self.r_sep)

        # init Q_exe state space
        self.time_state = [0] * self.M
        self.up_state = [0] * self.M
        self.computing_state = [0] * self.M
        self.down_state = [0] * self.M

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.M

        # Normalized norm
        self.T_value = []
        self.T_norm = 0
        self.D_up_value = []
        self.D_up_norm = 0
        self.D_computing_value = []
        self.D_computing_norm = 0
        self.D_download_value = []
        self.D_download_norm = 0
        for i in range(self.N):
            self.T_value.append(self.task[i].delay_threshold)
            self.D_up_value.append(self.task[i].upload_data)
            self.D_computing_value.append(self.task[i].computation_data)
            self.D_download_value.append(self.task[i].download_data)
        self.T_norm = max(self.T_value)
        self.D_up_norm = max(self.D_up_value)
        self.D_computing_norm = max(self.D_computing_value)
        self.D_download_norm = max(self.D_download_value)

        self.task_service_delay = []

    def reset_state(self):
        self.task = Task_generation(num_task=self.N, num_timeslots=self.T, compression_coefficient=self.r_sep)
        # init Q_exe state space
        self.time_state = [0] * self.M
        self.up_state = [0] * self.M
        self.computing_state = [0] * self.M
        self.down_state = [0] * self.M

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.M

        self.Task_done = [0] * self.N
        self.done = False  # 所有任务完成则为1
        self.task_service_delay = []
        state = self.up_state + self.computing_state + self.down_state + self.time_state
        state = np.array(state)

    def get_task_queue(self, time_slot):
        """
        Define waiting queues Q_wait and execution queues Q_exe
        Q_wait length: infinite
        Q_exe length: self.M
        :return:
        """
        for i in range(self.N):
            # if t = t_i then add req_i into Q_wait
            if time_slot == self.task[i].arrival_time:
                self.Q_wait.append(i + 1)  # 1  3  5 ...
                self.task[i].waiting_phase = 1
                # print('当前时隙到达任务：第{}号任务，已将其加入了任务等待队列当中'.format(i + 1))
        # print('当前任务等待队列:{}'.format(self.Q_wait))

        # init Q_wait_priority
        self.Q_wait_priority = []
        # init Q_wait queue state and get Q_wait queue task priority
        for j in range(len(self.Q_wait)):
            s_u = self.task[self.Q_wait[j] - 1].upload_data
            s_c = self.task[self.Q_wait[j] - 1].computation_data
            s_d = self.task[self.Q_wait[j] - 1].download_data
            # todo s_T 算的有问题 2023/9/20
            # 不等于表明其已经在队列等待了
            if self.task[self.Q_wait[j] - 1].rest_available_time != self.task[self.Q_wait[j] - 1].delay_threshold:
                s_T = self.task[self.Q_wait[j] - 1].rest_available_time
                # s_T = (self.task[self.Q_wait[j] - 1].arrival_time + self.task[self.Q_wait[j] - 1].rest_available_time)
            else:
                # 否则任务是刚进到等待队列中的
                # s_T = (self.task[self.Q_wait[j] - 1].arrival_time + self.task[self.Q_wait[j] - 1].delay_threshold)
                s_T = self.task[self.Q_wait[j] - 1].delay_threshold

            # calculate the priority weight
            self.Q_wait_priority.append((s_u + s_c + s_d) / 3 / s_T)  # todo S_T可能等于0，2023/10/2

        while True:
            is_available_Q_exe = 0
            if 0 in self.Q_exe:
                is_available_Q_exe = 1
            # 卸载两个条件：1.任务等待队列中有任务 2.边缘服务器运行队列有空闲位置
            if is_available_Q_exe == 1 and len(self.Q_wait) != 0:
                # 根据优先级获取要卸载的任务 系统id *
                offloading_task_id = self.Q_wait[self.Q_wait_priority.index(max(self.Q_wait_priority))]
                # 获取Q_exe queue第一个空闲位置的索引，并将任务添加到该位置
                free_index = self.Q_exe.index(0)
                self.Q_exe[free_index] = offloading_task_id  # 将卸载任务的系统id放到执行队列的空闲位置
                # print('将第{}号任务卸载到边缘服务器,并将其从任务等待队列当中删除'.format(offloading_task_id))
                # 修改任务阶段
                self.task[offloading_task_id - 1].waiting_phase = 0
                self.task[offloading_task_id - 1].upload_phase = 1

                # 初始化‘刚’卸载到执行队列中任务的状态
                self.time_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].rest_available_time
                self.up_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].upload_data
                self.computing_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].computation_data
                self.down_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].download_data

                # 将卸载完的该任务从等待队列Q_wait中移除，同时也从任务等待优先级队列Q_wait_priority中移除
                self.Q_wait.remove(offloading_task_id)
                self.Q_wait_priority.remove(self.Q_wait_priority[self.Q_wait_priority.index(max(self.Q_wait_priority))])
                # print('当前任务等待队列{}'.format(self.Q_wait))
            else:
                # print('任务等待队列中的任务已全部卸载完成，卸载结束！')
                break

        for i in range(len(self.Q_wait)):
            self.task[self.Q_wait[i] - 1].rest_available_time -= self.tao
            # todo 如果等待的时间超出了任务的延迟阈值，则需要直接从等待队列中移除 2023/10/2
            # if self.task[self.Q_wait[i] - 1].rest_available_time <= 0:
            #     self.Q_wait.remove(self.Q_wait[i])
            #     self.Q_wait_priority.remove(self.Q_wait_priority[i])

        Q_exe_state = self.up_state + self.computing_state + self.down_state + self.time_state
        return Q_exe_state

    def step(self, action, step):
        self.P_i = 30  # Tx power of vehicles: 30 dBm
        self.P_RSU = 27  # Tx power of RSU: 27 dBm

        up_B_ratio = action[:5]
        computing_resource_ratio = action[5:10]
        down_B_ratio = action[-5:]

        reward = 0
        up_rate = transmission_rate(len(action[:5]), self.P_i, up_B_ratio, self.R_u)
        computing_rate = []
        for i in range(len(computing_resource_ratio)):
            computing_rate.append(computing_resource_ratio[i] * self.R_c * 1e3 / 1e3)  # self.R_c * 1e3 => MHz,即Mcycles/s, 除以1000 => Mcycles/ms
        down_rate = transmission_rate(len(action[:5]), self.P_RSU, down_B_ratio, self.R_d)

        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] != 0:
                self.task[self.Q_exe[i] - 1].rest_available_time -= self.tao
                self.time_state[i] = self.task[self.Q_exe[i] - 1].rest_available_time
                # up_rate: Mbits/ms
                self.up_state[i] -= self.task[self.Q_exe[i] - 1].upload_phase * self.tao * up_rate[i]
                # computing_rate: Mcycles/ms
                self.computing_state[i] -= self.task[self.Q_exe[i] - 1].computing_phase * self.tao * computing_rate[i]
                self.down_state[i] -= self.task[self.Q_exe[i] - 1].download_phase * self.tao * down_rate[i]

                # 如果任务i的时间状态小于等于0 并且 下载状态大于0，任务直接未完成
                if self.time_state[i] <= 0 and self.down_state[i] > 0:
                    # print(f'第{self.Q_exe[i]}个任务未完成，已将Q_exe中移除-------------------------')
                    self.Task_done[self.Q_exe[i] - 1] = -1
                    self.task_service_delay.append(self.task[self.Q_exe[i] - 1].delay_threshold)
                    self.Q_exe[i] = 0
                    self.time_state[i] = 0
                    self.up_state[i] = 0
                    self.computing_state[i] = 0
                    self.down_state[i] = 0
                else:
                    if self.up_state[i] <= 0:
                        self.up_state[i] = 0
                        self.task[self.Q_exe[i] - 1].upload_phase = 0
                        self.task[self.Q_exe[i] - 1].computing_phase = 1

                    if self.computing_state[i] <= 0:
                        self.computing_state[i] = 0
                        self.task[self.Q_exe[i] - 1].computing_phase = 0
                        self.task[self.Q_exe[i] - 1].download_phase = 1

                    if self.down_state[i] <= 0:
                        self.task_service_delay.append(self.task[self.Q_exe[i] - 1].delay_threshold - self.task[self.Q_exe[i] - 1].rest_available_time)
                        self.down_state[i] = 0
                        self.time_state[i] = 0
                        self.task[self.Q_exe[i] - 1].download_phase = 0
                        # print(f"第{self.Q_exe[i]}个任务已完成")
                        self.Task_done[self.Q_exe[i] - 1] = 1
                        self.Q_exe[i] = 0

        new_state = self.up_state + self.computing_state + self.down_state + self.time_state
        while 0 not in self.Task_done:
            self.done = True
            break

        return new_state, reward, self.done


if __name__ == '__main__':
    num_tasks = 10
    num_timeslots = 20
    episodes = 1
    env = env(num_tasks, num_timeslots)
    for i in range(episodes):
        sum_reward = 0
        env.reset_state()
        for t in range(num_timeslots):
            print(f'------第{i}个episode，第{t}个时隙------')
            # obtain two queues
            env.get_task_queue(time_slot=t)
            # generate corresponding action, dimension: 3 * M
            # action = agent.choose_action(state)
            action = [20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10]
            new_state, reward, done = env.step(action)
            if done:
                break
            # sum_reward += reward
            # state = new_state
