import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# CSVファイルからデータを読み取る関数
def read_csv_data(csv_file_path):
    time_data = []
    theta_data = []
    omega_data = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # ヘッダー行をスキップ
        for row in csv_reader:
            time_data.append(float(row[0]))
            theta_data.append(float(row[1]))
            omega_data.append(float(row[2]))

    return time_data, theta_data, omega_data

# アニメーションの初期化関数
def init():
    pendulum.set_data([], [])
    return pendulum,

# アニメーションのフレーム更新関数
def update(frame):
    x = np.sin(theta_data[frame])  # 振り子のx座標
    y = -np.cos(theta_data[frame])  # 振り子のy座標
    pendulum.set_data([0, x], [0, y])  # 振り子の位置を更新
    return pendulum,

# CSVファイルのパス
csv_file_path = 'hougan_single_pendulum_rl_data_epoch_1.csv'

# CSVファイルからデータを読み取る
time_data, theta_data, omega_data = read_csv_data(csv_file_path)

# アニメーションの設定
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
pendulum, = ax.plot([], [], lw=2)

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(time_data), init_func=init, blit=True, interval=2)

# アニメーションの保存
save_path = 'hougan_single_pendulum_rl_data_epoch_1.gif'
ani.save(save_path, writer='pillow', fps=30)
print(f'Animation saved to {save_path}')

plt.show()
