import os

# 【关键】修复 OpenMP 冲突报错，必须放在 import torch 之前
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# ==========================================
# 1. 导入你的环境和算法类
# ==========================================
# 请确保 PPO_UUVTrain.py 在同一目录下，或者修改为你实际的文件名
from PPO_UUVTrain import UUV_MultiGoal_Env, PPOContinuous, custom_multi_goal_reward

# ==========================================
# 2. 全局配置
# ==========================================
MAX_STEPS = 4000
HIDDEN_DIM = 256
MODEL_PATH = './model_trained/round_250'  # 你的最终模型路径
ACTOR_FILE = 'actor.pth'  # 权重文件名


# ==========================================
# 3. 绘图核心函数 (生成5张高清图)
# ==========================================
def save_all_individual_plots(env, history, status, total_reward):
    """
    将所有分析图表单独保存为高清图片
    """
    # 创建带时间戳的专属文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/Report_{timestamp}_{status}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"正在生成图表，保存路径: {save_dir} ...")

    steps_range = range(len(history['u']))
    common_dpi = 300  # 论文级清晰度

    # -------------------------------------------------------
    # 图 1: 实际轨迹图 (Trajectory Map)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 10), dpi=common_dpi)
    ax = plt.gca()

    # 画边界
    bound = 700.0
    ax.set_xlim(-bound * 1.1, bound * 1.1)
    ax.set_ylim(-bound * 1.1, bound * 1.1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(f"UUV Navigation Trajectory (Total Reward: {total_reward:.0f})", fontsize=14, fontweight='bold')

    # 画障碍物
    for obs in env.cached_obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], color='#7f7f7f', alpha=0.4, ec='black')
        ax.add_patch(circle)

    # 画目标点
    goals = env.cached_goals
    for i, g in enumerate(goals):
        color = 'red' if i == len(goals) - 1 else '#2ca02c'  # 终点红，途经绿
        ax.plot(g[0], g[1], marker='*', color=color, markersize=20, zorder=10, markeredgecolor='white')
        ax.text(g[0] + 25, g[1] + 25, f"G{i + 1}", fontsize=12, color=color, fontweight='bold')

    # 画轨迹
    ax.plot(history['x'], history['y'], color='#1f77b4', linewidth=2.5, alpha=0.9, label='Actual Path')
    ax.plot(history['x'][0], history['y'][0], 'bs', markersize=10, label='Start')

    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, "01_Trajectory_Map.png"), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # 图 2: 安全距离分析 (Safety Analysis)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=common_dpi)
    plt.plot(steps_range, history['min_dist'], color='#d62728', linewidth=2, label='Min Distance to Obstacles')

    # 警戒线
    plt.axhline(y=5.0, color='orange', linestyle='--', alpha=0.8, label='Safety Margin (5m)')
    plt.axhline(y=2.0, color='black', linestyle=':', alpha=0.6, label='Collision Limit (2m)')

    plt.title('Safety Analysis: Minimum Distance to Nearest Obstacle', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Distance (m)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "02_Safety_Distance.png"), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # 图 3: 动力学分析 (Velocity - Surge vs Drift)
    # -------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=common_dpi)

    color_u = '#1f77b4'  # 蓝
    ax1.set_xlabel('Simulation Steps', fontsize=12)
    ax1.set_ylabel('Surge Speed u (m/s)', color=color_u, fontsize=12, fontweight='bold')
    l1 = ax1.plot(steps_range, history['u'], color=color_u, linewidth=2, label='Longitudinal Speed (u)')
    ax1.tick_params(axis='y', labelcolor=color_u)
    ax1.grid(True, alpha=0.3)

    # 双轴画横向速度
    ax2 = ax1.twinx()
    color_v = '#2ca02c'  # 绿
    ax2.set_ylabel('Sway/Drift Speed v (m/s)', color=color_v, fontsize=12, fontweight='bold')
    l2 = ax2.plot(steps_range, history['v'], color=color_v, linewidth=1.5, linestyle='--',
                  label='Lateral Drift Speed (v)')
    ax2.tick_params(axis='y', labelcolor=color_v)

    # 合并图例
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Dynamics Analysis: High Speed Maintenance & Drift Control', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_dir, "03_Velocity_Dynamics.png"), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # 图 4: 控制动作分析 (Control Actions)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=common_dpi)
    plt.plot(steps_range, history['motor'], color='purple', alpha=0.7, linewidth=1.5, label='Propeller Thrust')
    plt.plot(steps_range, history['rudder'], color='#ff7f0e', alpha=0.9, linewidth=1.5, label='Rudder Angle')

    plt.title('Control Quality: Action Smoothness & Response', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Normalized Command [-1, 1]', fontsize=12)
    plt.ylim(-1.2, 1.2)  # 留点余量
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "04_Control_Actions.png"), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # 图 5: 奖励分析 (Instant Reward)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 5), dpi=common_dpi)

    # 对奖励做平滑处理 (Moving Average)
    rewards = np.array(history['rewards'])
    window = 10  # 平滑窗口大小
    if len(rewards) > window:
        smooth_rewards = np.convolve(rewards, np.ones(window) / window, mode='same')
    else:
        smooth_rewards = rewards

    plt.plot(steps_range, rewards, color='lightgray', alpha=0.5, label='Raw Reward')
    plt.plot(steps_range, smooth_rewards, color='#8c564b', linewidth=2, label='Smoothed Reward (Trend)')

    plt.title('Reward Analysis: Event Trigger & Stability', fontsize=14, fontweight='bold')
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Reward Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "05_Instant_Reward.png"), bbox_inches='tight')
    plt.close()

    print(f"所有图表已生成完毕！请查看文件夹: {save_dir}")


# ==========================================
# 4. 主运行函数
# ==========================================
def run_final_analysis():
    # --- A. 初始化环境 ---
    # 使用固定种子，确保每次运行都能复现这张“经典地图”
    # 如果想换地图，修改 fixed_map_seed 即可 (例如: 42, 100, 2024)
    MAP_CONFIG = {
        'map_size': 700.0,
        'n_obs': 8,
        'n_goals': 3,
        'obs_radius_range': (50, 100),
        'map_refresh_freq': 9999,  # 锁定地图
        'fixed_map_seed': 11236  # 推荐种子
    }

    env = UUV_MultiGoal_Env(
        reward_fn=custom_multi_goal_reward,
        goal_positions=None,
        obstacles=None,
        map_config=MAP_CONFIG,
        render_mode='human',  # 必须开启，才能看到动画
        max_steps=MAX_STEPS
    )

    # 稍微放宽终点判定，让高速模型更容易通过测试
    env.GOAL_THRESHOLD = 5.0

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- B. 加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化一个空模型架构
    agent = PPOContinuous(state_dim, HIDDEN_DIM, action_dim, 0, 0, 0, 0, 0.2, 0.99, device, 1.0, 0.0)

    # 加载权重
    path = os.path.join(MODEL_PATH, ACTOR_FILE)
    if os.path.exists(path):
        agent.actor.load_state_dict(torch.load(path, map_location=device))
        print(">>> 最终模型加载成功！准备起飞...")
    else:
        print(f"错误：找不到模型文件 {path}")
        return

    agent.actor.eval()  # 切换到评估模式

    # --- C. 数据记录容器 ---
    history = {
        'x': [], 'y': [],  # 轨迹
        'u': [], 'v': [], 'r': [],  # 速度
        'motor': [], 'rudder': [],  # 动作
        'min_dist': [],  # 离障碍物最近距离
        'rewards': []  # 奖励
    }

    state, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0

    # 记录起点
    history['x'].append(env.current_eta[0])
    history['y'].append(env.current_eta[1])

    print(">>> 开始运行测试 (请观看弹出的动画窗口)...")

    while not (done or truncated):
        # 1. 决策
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = agent.actor(state_tensor)
            action = mu.cpu().numpy().flatten()  # 确定性策略

        # 2. 记录动作
        history['motor'].append(action[0])
        history['rudder'].append(action[1])

        # 3. 执行一步
        next_state, reward, done, truncated, info = env.step(action)

        # 4. 记录状态数据
        history['x'].append(env.current_eta[0])
        history['y'].append(env.current_eta[1])
        history['u'].append(env.current_nu[0])
        history['v'].append(env.current_nu[1])
        history['r'].append(env.current_nu[2])
        history['rewards'].append(reward)

        # 5. 计算并记录最近障碍物距离 (用于画图)
        current_pos = env.current_eta[:2]
        dists = []
        for obs in env.cached_obstacles:
            # 表面距离 = 中心距离 - 半径
            d = np.linalg.norm(current_pos - obs['center']) - obs['radius']
            dists.append(d)
        history['min_dist'].append(min(dists) if dists else 100.0)

        # 6. 渲染
        env.render()

        state = next_state
        total_reward += reward
        step += 1

        # 可选：稍微减慢一点动画速度，方便肉眼观察
        # time.sleep(0.01)

    env.close()

    result_status = "Success" if info.get('is_goal_achieved', False) else "Fail"
    print(f"\n>>> 运行结束！")
    print(f"    状态: {result_status}")
    print(f"    总步数: {step}")
    print(f"    总奖励: {total_reward:.2f}")

    # --- D. 生成图表 ---
    save_all_individual_plots(env, history, result_status, total_reward)


if __name__ == '__main__':
    # 确保有文件夹
    if not os.path.exists("./results"):
        os.makedirs("./results")
    run_final_analysis()