import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium
from gymnasium.vector import AsyncVectorEnv

# --- 导入自定义模块 ---
# 请确保这两个文件在同一目录下
from PPO_Algorithm import PPOContinuous, PPOBuffer, get_log_prob_batch
from UUV_Env import UUV_MultiGoal_Env, custom_multi_goal_reward
import rl_utils  # 假设你有一个简单的 moving_average 函数，如果没有，我在下面提供了一个简单的实现


# ==========================================
# 0. 简单的移动平均工具 (如果你没有 rl_utils)
# ==========================================
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ==========================================
# 1. 超参数配置
# ==========================================
# --- 环境设置 ---
NUM_ENVS = 16  # 并行环境数量 (CPU核心数越多越好)
MAX_STEPS = 1000  # 单回合最大步数 (给足够的时间到达所有目标)
GOAL_POSITIONS = [(300, 100), (-100, 200), (0, -400)]
OBSTACLES = []  # (x, y, r)

# --- 训练设置 ---
TOTAL_EPISODES = 8000  # 总训练回合数 (用于控制总时长)
STEPS_PER_UPDATE = 2048  # 每次 PPO 更新收集的步数 (必须是 NUM_ENVS 的倍数)
REWARD_SCALE = 100.0  # 奖励缩放分母 (用于日志显示，PPO内部可能有自己的缩放)

# --- PPO 超参数 ---
ACTOR_LR = 1e-5
CRITIC_LR = 5e-6
HIDDEN_DIM = 256
GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 10
EPS_CLIP = 0.2
ENT_COEF = 0.01  # 熵系数

# --- 保存与评估 ---
SAVE_INTERVAL = 50  # 每多少次 Update 保存一次模型
EVAL_INTERVAL = 20  # 每多少次 Update 进行一次可视化评估
MODEL_DIR = './model_save'


# ==========================================
# 2. 辅助函数
# ==========================================

def make_env(rank, render_mode='none'):
    """
    创建环境的工厂函数，用于 AsyncVectorEnv
    rank: 环境的索引，用于设置不同的随机种子
    """

    def _init():
        env = UUV_MultiGoal_Env(
            GOAL_POSITIONS,
            custom_multi_goal_reward,
            OBSTACLES,
            render_mode=render_mode,
            max_steps=MAX_STEPS
        )
        # 为每个环境设置不同的种子，防止并行环境运行一模一样
        env.reset(seed=rank + 1000)
        return env

    return _init


def evaluate_policy(agent, device, render=False):
    """
    评估函数：暂停并行训练，使用单环境测试当前策略
    render: 是否开启可视化窗口
    """
    # 创建一个独立的评估环境
    env = UUV_MultiGoal_Env(GOAL_POSITIONS, custom_multi_goal_reward, OBSTACLES,
                            render_mode='human' if render else 'none', max_steps=MAX_STEPS)

    state, _ = env.reset(seed=42)  # 固定种子以便对比
    done = False
    total_reward = 0
    steps = 0

    while not done:
        if render:
            env.render()

        # 准备状态数据：(State_Dim,) -> (1, State_Dim)
        state_tensor = np.array([state])

        # 获取动作 (使用 take_action)
        # 注意：take_action 返回 batch 形式，我们需要取 [0]
        # 评估时我们不需要计算梯度，也不需要探索噪声，但这里复用 take_action 采样也可以
        # 如果想要确定性策略，可以修改 Agent 增加 deterministic 模式，或者直接用 mean
        with torch.no_grad():
            _, action_phys = agent.take_action(state_tensor)

        # 环境执行物理动作
        action = action_phys[0]
        next_state, reward, term, trunc, _ = env.step(action)

        done = term or trunc
        total_reward += reward
        state = next_state
        steps += 1

    env.close()
    return total_reward, steps


# ==========================================
# 3. 主程序
# ==========================================
# ================================================================
# 用于直接运行本文件进行训练的主程序 (逻辑与 main.py 一致)
# ================================================================
if __name__ == '__main__':
    import os
    import gymnasium
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from gymnasium.vector import AsyncVectorEnv

    # 引入环境 (PPO_Algorithm.py 头部已经引入了部分宏定义，这里补充引入 Env 类)
    # 确保 UUV_Env.py 在同一目录下
    try:
        from UUV_Env import UUV_MultiGoal_Env, custom_multi_goal_reward
    except ImportError:
        print("❌ 错误: 找不到 UUV_Env.py，无法运行完整训练循环。")
        exit()


    # --- 0. 辅助函数: 移动平均 ---
    def moving_average(a, window_size):
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size - 1, 2)
        begin = np.cumsum(a[:window_size - 1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))


    # --- 1. 超参数配置 ---
    # 环境配置
    NUM_ENVS = 16  # 并行环境数量
    MAX_STEPS = 700  # 单回合最大步数
    GOAL_POSITIONS = [(300, 100), (-100, 200), (0, -400)]
    OBSTACLES = []

    # 训练配置
    TOTAL_EPISODES = 5000  # 总训练回合数
    STEPS_PER_UPDATE = 2048  # 每次更新收集的步数
    REWARD_SCALE = 100.0  # 奖励缩放
    REWARD_SHIFT = 0.0

    # PPO 参数
    ACTOR_LR = 1e-5
    CRITIC_LR = 5e-6
    HIDDEN_DIM = 256
    GAMMA = 0.99
    LAMBDA = 0.95
    EPOCHS = 10
    EPS_CLIP = 0.2
    ENT_COEF = 0.01

    # 保存路径
    MODEL_DIR = './model_save_algo_run'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


    # --- 2. 环境工厂函数 ---
    def make_env(rank):
        def _init():
            env = UUV_MultiGoal_Env(
                GOAL_POSITIONS,
                custom_multi_goal_reward,
                OBSTACLES,
                render_mode='none',
                max_steps=MAX_STEPS
            )
            env.reset(seed=rank + 2000)
            return env

        return _init


    # --- 3. 初始化 ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"🚀 [PPO_Algorithm Direct Run] Start on {device} with {NUM_ENVS} environments.")

    # 创建并行环境
    uuv_env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])

    state_dim = uuv_env.single_observation_space.shape[0]
    action_dim = uuv_env.single_action_space.shape[0]

    # 初始化 Agent
    agent = PPOContinuous(state_dim, HIDDEN_DIM, action_dim,
                          ACTOR_LR, CRITIC_LR, LAMBDA, EPOCHS, EPS_CLIP, GAMMA,
                          device, ent_coef=ENT_COEF)

    buffer = PPOBuffer()

    # 变量记录
    return_list = []
    episode_returns = np.zeros(NUM_ENVS)
    finished_episodes = 0
    total_steps = 0

    # 初始状态
    states, _ = uuv_env.reset()
    pbar = tqdm(total=TOTAL_EPISODES, desc="Training")

    try:
        while finished_episodes < TOTAL_EPISODES:

            # === 阶段 1: 数据收集 ===
            for _ in range(STEPS_PER_UPDATE // NUM_ENVS):

                # 获取动作 (Norm 用于训练, Phys 用于执行)
                action_norm, action_phys = agent.take_action(states)

                # 计算旧策略 LogProb
                log_probs = get_log_prob_batch(agent, states, action_norm)

                # 环境步进
                next_states, rewards, terminations, truncations, infos = uuv_env.step(action_phys)
                dones = terminations | truncations

                # 存入 Buffer
                buffer.push(states, action_norm, rewards, next_states, dones, log_probs)

                # 记录奖励
                episode_returns += rewards

                # 处理结束的回合
                for i, done in enumerate(dones):
                    if done:
                        return_list.append(episode_returns[i] / REWARD_SCALE)
                        episode_returns[i] = 0
                        finished_episodes += 1
                        pbar.update(1)

                states = next_states
                total_steps += NUM_ENVS

            # === 阶段 2: 更新 ===
            transition_dict = buffer.get_data()
            agent.update(transition_dict)
            buffer.clear()

            # 显示进度
            if len(return_list) > 0:
                pbar.set_postfix({'AvgRet': f'{np.mean(return_list[-10:]):.2f}'})

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted.")
    finally:
        pbar.close()
        uuv_env.close()

    # --- 4. 绘图 ---
    print("Drawing training curve...")
    plt.figure(figsize=(12, 6))
    plt.plot(return_list, alpha=0.3, color='gray', label='Raw')
    if len(return_list) > 10:
        mv_return = moving_average(return_list, 19)
        plt.plot(mv_return, color='red', linewidth=2, label='Moving Avg')
    plt.title("PPO Training Curve (Algorithm File Run)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'algo_run_curve.png'))
    plt.show()
    print("Done.")