# under different r_sep, collaborative sensing tasks performance
import matplotlib.pyplot as plt
import numpy as np


def service_time():
    # number tasks = 10, r_sep 0.2    0.4      0.6    0.8      1.0
    proposed_scheme_1_time = [98.75, 105.79, 111.77, 127.16, 139.90]
    # number tasks = 20
    proposed_scheme_2_time = [102.58, 114.72, 130.41, 159.75, 193.31]
    # number tasks = 30
    proposed_scheme_3_time = [132.51, 161.03, 182.92, 225.18, 252.81]
    # number tasks = 40
    proposed_scheme_4_time = [182.41, 218.02, 237.19, 264.21, 274.65]
    # number tasks = 50
    proposed_scheme_5_time = [232.94, 251.95, 264.28, 277.91, 284.72]

    x = np.arange(0.2, 1.2, 0.2)
    y = np.arange(75, 350, 25)
    fig, ax = plt.subplots()
    plt.ylim((75, 325))
    ax.plot(x, proposed_scheme_5_time, 'k<-', linewidth=1, label='N=50')
    ax.plot(x, proposed_scheme_4_time, 'ms-', linewidth=1, label='N=40')
    ax.plot(x, proposed_scheme_3_time, 'bx-', linewidth=1, label='N=30')
    ax.plot(x, proposed_scheme_2_time, 'g*-', linewidth=1, label='N=20')
    ax.plot(x, proposed_scheme_1_time, 'yd-', linewidth=1, label='N=10')

    plt.xticks(x)
    plt.yticks(y)

    plt.grid()
    plt.legend(loc='upper left', prop={'size': 10})  # 显示标签
    plt.xlabel('r$_{sep}$', fontsize=14)  # 设置X轴名称
    plt.ylabel('Average service time (ms)', fontsize=12)  # 设置Y轴名称
    # plt.savefig('Fig5', dpi=1200, bbox_inches='tight')
    plt.show()


def completion_rate():
    # number tasks = 10, r_sep 0.2    0.4      0.6    0.8      1.0
    proposed_scheme_1 = [0.967, 0.968, 0.965, 0.940, 0.907]
    # number tasks = 20
    proposed_scheme_2 = [0.931, 0.909, 0.885, 0.785, 0.686]
    # number tasks = 30
    proposed_scheme_3 = [0.882, 0.847, 0.792, 0.617, 0.455]
    # number tasks = 40
    proposed_scheme_4 = [0.827, 0.723, 0.614, 0.384, 0.287]
    # number tasks = 50
    proposed_scheme_5 = [0.704, 0.548, 0.412, 0.260, 0.182]

    x = np.arange(0.2, 1.2, 0.2)
    # y = np.arange(75, 350, 50)
    fig, ax = plt.subplots()
    # plt.ylim((75, 325))
    ax.plot(x, proposed_scheme_1, 'yd-', linewidth=1, label='N=10')
    ax.plot(x, proposed_scheme_2, 'g*-', linewidth=1, label='N=20')
    ax.plot(x, proposed_scheme_3, 'bx-', linewidth=1, label='N=30')
    ax.plot(x, proposed_scheme_4, 'ms-', linewidth=1, label='N=40')
    ax.plot(x, proposed_scheme_5, 'k<-', linewidth=1, label='N=50')

    plt.xticks(x)
    # plt.yticks(y)

    plt.grid()
    plt.legend(loc='lower left', prop={'size': 12})  # 显示标签
    plt.xlabel('r$_{sep}$', fontsize=14)  # 设置X轴名称
    plt.ylabel('Average completion rate', fontsize=13)  # 设置Y轴名称
    # plt.savefig('Fig5', dpi=1200, bbox_inches='tight')
    plt.show()


service_time()
# completion_rate()
