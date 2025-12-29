import random
from typing import Dict

import gymnasium
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import collections
from gymnasium.vector import SyncVectorEnv
import gymnasium.vector
from gymnasium.vector import AsyncVectorEnv

try:
    from UUV_Env import UUV_MultiGoal_Env, custom_multi_goal_reward, \
        MAX_RUDDER_ANGLE, MAX_MOTOR_SPEED, SIM_BOUND, DT, R_COLLISION, R_FINAL_GOAL, R_STAGE_GOAL
except ImportError:
    print("⚠️ 警告: 未找到 UUV_Env.py，请确保文件在同一目录下。")
    exit()

# --- 配置 ---
RL_MODE = 'trainmode'
PTH_LOADED = True
MODEL_PATH = './model_trained/round_250'
ACTOR_FILE = 'actor.pth'
CRITIC_FILE = 'critic.pth'
# 并行环境数量
NUM_ENVS = 16


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        # 均值输出层
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        # 标准差输出层
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 使用 tanh 激活函数，与 PPO 常见设置保持一致
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        # mu 输出范围 (-1, 1)
        mu = torch.tanh(self.fc_mu(x))
        # std 必须是正数
        std = F.softplus(self.fc_std(x)) + 1e-5

        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 保持激活函数一致性
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, reward_scale, ent_coef=0.01):

        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.reward_scale = reward_scale  # 奖励缩放系数
        self.ent_coef = ent_coef

    def take_action(self, state_batch: np.ndarray):
        """
        输入: [NUM_ENVS, state_dim]
        输出:
            action_norm_np: 用于训练和环境输入 ([-1, 1])
            action_physical: 也是 action_norm_np (因为现在的 Env 会自己做物理映射)
        """
        state_tensor = torch.tensor(state_batch, dtype=torch.float).to(self.device)

        with torch.no_grad():
            mu, std = self.actor(state_tensor)
            action_dist = torch.distributions.Normal(mu, std)
            action_normalized = action_dist.sample()
            # 严格限制在 [-1, 1]
            action_normalized = torch.clamp(action_normalized, -1.0, 1.0)

        action_norm_np = action_normalized.cpu().numpy()

        # 【关键修改】
        # 因为 UUV_Env.step() 里面已经写了: motor = (action + 1)/2 * MAX
        # 所以这里传给环境的 action_physical 直接就是 normalized 的值即可！
        # 不要在这里乘 70 或者 0.52，否则会重复缩放导致数值爆炸。
        action_physical = action_norm_np

        return action_norm_np, action_physical

    def update(self, transition_dict):
        # 1. 数据准备 [Steps, Envs, Dim]
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).unsqueeze(-1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).unsqueeze(-1)
        old_log_probs = torch.tensor(transition_dict['log_probs'], dtype=torch.float).to(self.device)

        # 奖励缩放
        rewards = rewards / self.reward_scale

        # 2. 计算 GAE (在并行数据展平之前计算)
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

            delta = rewards + self.gamma * next_values * (1 - dones) - values

            advantage = torch.zeros_like(delta).to(self.device)
            gae = 0
            # 从最后一步往前推
            for t in reversed(range(states.shape[0])):
                gae = delta[t] + self.gamma * self.lmbda * gae * (1 - dones[t])
                advantage[t] = gae

        # 3. 数据展平 [Steps * Envs, Dim]
        states = states.view(-1, states.shape[-1])
        actions = actions.view(-1, self.action_dim)
        old_log_probs = old_log_probs.view(-1, 1)
        advantage = advantage.view(-1, 1)
        values = values.view(-1, 1)
        td_target = advantage + values  # Value Target

        # Advantage 归一化
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        total_samples = states.size(0)
        batch_size = 512  # 建议: 256 或 512
        for _ in range(self.epochs):
            # 每次 Epoch 都打乱索引
            indices = torch.randperm(total_samples)
            # 遍历每一个 Mini-batch
            for i in range(0, total_samples, batch_size):
                # 提取当前 Batch 的索引
                batch_indices = indices[i: i + batch_size]

                # 取出数据
                bs = states[batch_indices]
                ba = actions[batch_indices]
                bolp = old_log_probs[batch_indices]
                badv = advantage[batch_indices]
                btd = td_target[batch_indices]

                # 计算 Actor Loss
                mu, std = self.actor(bs)
                action_dists = Normal(mu, std)
                log_probs = action_dists.log_prob(ba).sum(dim=1, keepdim=True)
                entropy = action_dists.entropy().sum(dim=1, keepdim=True).mean()

                ratio = torch.exp(log_probs - bolp)
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * badv
                actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.ent_coef * entropy

                # 计算 Critic Loss
                critic_loss = torch.mean(F.mse_loss(self.critic(bs), btd.detach()))

                total_loss = actor_loss + critic_loss * 0.5

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪 (防止爆炸)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def save_model(self, path):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        print(f"\nModel parameters saved to: {path}")


class PPOBuffer:
    def __init__(self):
        self.transitions = collections.defaultdict(list)

    def push(self, state, action, reward, next_state, done, log_prob):
        self.transitions['states'].append(state)
        self.transitions['actions'].append(action)
        self.transitions['rewards'].append(reward)
        self.transitions['next_states'].append(next_state)
        self.transitions['dones'].append(done)
        self.transitions['log_probs'].append(log_prob)

    def get_data(self) -> Dict[str, np.ndarray]:
        data = {}
        for key, value_list in self.transitions.items():
            data[key] = np.array(value_list, dtype=np.float32)
        return data

    def clear(self):
        self.transitions.clear()


def get_log_prob_batch(agent, states, actions):
    with torch.no_grad():
        state_tensor = torch.tensor(states, dtype=torch.float).to(agent.device)
        action_tensor = torch.tensor(actions, dtype=torch.float).to(agent.device)
        mu, std = agent.actor(state_tensor)
        action_dists = Normal(mu, std)
        log_probs = action_dists.log_prob(action_tensor).sum(dim=1).cpu().numpy()
        return log_probs


# ================================================================
# 主程序
# ================================================================
if __name__ == '__main__':
    import os
    import numpy as np
    import torch
    from tqdm import tqdm
    from gymnasium.vector import AsyncVectorEnv


    # 假设你这些类都在同一个文件或已导入
    # from your_module import UUV_MultiGoal_Env, PPOContinuous, PPOBuffer, get_log_prob_batch

    # 辅助函数: 移动平均
    def moving_average(a, window_size):
        if len(a) < window_size: return np.array(a)
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size - 1, 2)
        begin = np.cumsum(a[:window_size - 1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))


    # --- 1. 超参数配置 ---
    MAX_STEPS = 4000
    NUM_ENVS = 16  # 【新增】必须定义并行环境数量

    # 【修改】不再需要固定的 GOAL_POSITIONS 和 OBSTACLES
    # 定义地图生成配置
    MAP_CONFIG = {
        'map_size': 700.0,
        'n_obs': 8,  # 障碍物数量
        'n_goals': 3,  # 目标点数量
        'obs_radius_range': (50, 100),  # 半径范围
        'map_refresh_freq': 40  # 地图刷新频率
    }

    TOTAL_EPISODES = 10000  # 这个参数其实不控制循环，控制循环的是 TOTAL_ROUNDS

    # 【修改】每 80 个回合打印一次日志
    EPISODES_PER_LOG = 80

    # 总共跑多少轮日志
    TOTAL_ROUNDS = 250

    STEPS_PER_UPDATE = 16384

    # 【核心配置】奖励缩放
    REWARD_SCALE = 400.0  # 确认使用这个参数

    ACTOR_LR = 3e-5  #
    CRITIC_LR = 1e-4
    HIDDEN_DIM = 256
    GAMMA = 0.99
    LAMBDA = 0.95
    EPOCHS = 5
    EPS_CLIP = 0.1
    ENT_COEF = 0.001

    # 模型保存路径
    MODEL_DIR = './model_trained'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # --- 2. 环境构造函数 ---
    def make_env(rank):
        def _init():
            # 【核心修改】这里实例化环境时，传入 map_config，并且不传固定障碍物
            env = UUV_MultiGoal_Env(
                reward_fn=custom_multi_goal_reward,
                goal_positions=None,  # 设为 None 启用随机
                obstacles=None,  # 设为 None 启用随机
                map_config=MAP_CONFIG,  # 传入配置
                render_mode='none',
                max_steps=MAX_STEPS
            )
            # 这里的 seed 很重要，保证不同环境生成的随机地图不一样，增加多样性
            env.reset(seed=rank + 1000)
            return env

        return _init


    # --- 3. 初始化 ---
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[PPO] Start on {device} with {NUM_ENVS} environments.")

    # 创建并行环境
    uuv_env = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])

    state_dim = uuv_env.single_observation_space.shape[0]
    action_dim = uuv_env.single_action_space.shape[0]

    # 初始化 Agent
    agent = PPOContinuous(state_dim, HIDDEN_DIM, action_dim,
                          ACTOR_LR, CRITIC_LR, LAMBDA, EPOCHS, EPS_CLIP, GAMMA,
                          device, reward_scale=REWARD_SCALE, ent_coef=ENT_COEF)

    buffer = PPOBuffer()

    # 全局统计
    global_return_list = []

    # PPO Buffer 计数器
    current_buffer_steps = 0
    env_episode_returns = np.zeros(NUM_ENVS)

    states, _ = uuv_env.reset()

    # --- 加载模型逻辑 ---
    if PTH_LOADED:
        actor_full_path = os.path.join(MODEL_PATH, ACTOR_FILE)
        critic_full_path = os.path.join(MODEL_PATH, CRITIC_FILE)
        print(f"Loading pretrained model from: {MODEL_PATH} ...")

        if os.path.exists(actor_full_path) and os.path.exists(critic_full_path):
            # 加上 weights_only=True 可以消除警告，如果不兼容旧模型则改回 False
            agent.actor.load_state_dict(torch.load(actor_full_path, map_location=device, weights_only=True))
            agent.critic.load_state_dict(torch.load(critic_full_path, map_location=device, weights_only=True))
            print("Model Loaded Successfully.")
        else:
            print(f"Warning: Model files not found in {MODEL_PATH}, starting from scratch.")

    initial_actor_lr = ACTOR_LR
    initial_critic_lr = CRITIC_LR

    try:
        # === 外层循环：轮次 (Round) ===
        for i_round in range(TOTAL_ROUNDS):

            # 1. 学习率线性衰减 (Linear Decay)
            # 【修正】逻辑修正：i_round 从 0 开始，原来的公式在第0轮会算出一大于1的系数
            frac = 1.0 - (i_round / TOTAL_ROUNDS)
            new_actor_lr = max(initial_actor_lr * frac, 1e-6)
            new_critic_lr = max(initial_critic_lr * frac, 3e-6)

            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = new_actor_lr
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = new_critic_lr

            # 2. 本轮日志统计变量
            log_finished_episodes = 0
            log_rewards = []
            log_goals = 0  # 最终成功的次数 (Success Rate)
            log_stage_goals = 0  # 阶段性目标计数 (Hits)
            ppo_update_counts = 0

            # 3. 进度条
            with tqdm(total=EPISODES_PER_LOG, desc=f'Round {i_round + 1}/{TOTAL_ROUNDS}', unit='ep') as pbar:

                while log_finished_episodes < EPISODES_PER_LOG:

                    # --- A. 采样一步 ---
                    action_norm, action_phys = agent.take_action(states)

                    # 注意：get_log_prob_batch 需要你自己定义的函数
                    log_probs = get_log_prob_batch(agent, states, action_norm)

                    next_states, rewards, terminations, truncations, infos = uuv_env.step(action_phys)
                    dones = terminations | truncations

                    # --- 统计阶段性目标 (Hits) ---
                    # is_goal_achieved 在我们的 Env 中是指 "是否到达了当前的目标点" (不论是阶段还是最终)
                    # AsyncVectorEnv 的 infos 有时是字典，有时是元组，需要做兼容处理
                    if isinstance(infos, dict) and 'is_goal_achieved' in infos:
                        log_stage_goals += np.sum(infos['is_goal_achieved'])
                    elif isinstance(infos, tuple) or isinstance(infos, list):
                        # 处理有些版本的 gym vector env 返回 tuple 的情况
                        for info in infos:
                            if info.get('is_goal_achieved', False):
                                log_stage_goals += 1

                    buffer.push(states, action_norm, rewards, next_states, dones, log_probs)

                    current_buffer_steps += NUM_ENVS
                    env_episode_returns += rewards

                    # --- B. 处理结束的回合 ---
                    dones_idx = np.where(dones)[0]
                    num_dones = len(dones_idx)

                    if num_dones > 0:
                        for idx in dones_idx:
                            # 1. 记录分数
                            ep_ret = env_episode_returns[idx]
                            log_rewards.append(ep_ret)
                            global_return_list.append(ep_ret)
                            env_episode_returns[idx] = 0

                            # 2. 统计最终通关 Goal (Success)
                            # 在 AsyncVectorEnv 中，结束时的 info 会被包装在 final_info 里
                            is_success = False

                            # 写法兼容性处理
                            if 'final_info' in infos:
                                final_info = infos['final_info'][idx]
                                if final_info is not None and final_info.get('is_goal_achieved', False):
                                    is_success = True

                            # 你的 env 逻辑是：到达最后一个点，terminated=True, is_goal_achieved=True
                            # 所以只要 terminated 且 is_goal_achieved 就代表通关
                            # 但 vector env 中 terminations[idx] 是当前的 termination

                            # 简单判断：如果这是 termination 且 info 里说 goal reached
                            if is_success:
                                log_goals += 1

                        log_finished_episodes += num_dones
                        pbar.update(num_dones)

                        if len(log_rewards) > 0:
                            avg_ret = np.mean(log_rewards)
                            pbar.set_postfix({
                                'Avg': f'{avg_ret:.0f}',
                                'Succ': f'{log_goals}',  # 最终通关数
                                'Hits': f'{log_stage_goals}',  # 吃到的小球总数
                                'Upd': f'{ppo_update_counts}',
                            })

                    states = next_states

                    # --- C. PPO 更新 ---
                    if current_buffer_steps >= STEPS_PER_UPDATE:
                        transition_dict = buffer.get_data()
                        agent.update(transition_dict)
                        buffer.clear()
                        current_buffer_steps = 0
                        ppo_update_counts += 1

            # 保存模型
            if (i_round + 1) % 5 == 0:
                save_path = os.path.join(MODEL_DIR, f'round_{i_round + 1}')
                agent.save_model(save_path)
                print(
                    f" Round {i_round + 1} Summary: Avg={np.mean(log_rewards):.1f}, Success={log_goals}, Hits={log_stage_goals}")

    except KeyboardInterrupt:
        print("\n Training interrupted.")
    finally:
        uuv_env.close()

    # --- 4. 绘图 ---
    print("Drawing training curve...")
    plt.figure(figsize=(12, 6))
    plt.plot(global_return_list, alpha=0.3, color='gray', label='Raw')
    if len(global_return_list) > 20:
        mv_return = moving_average(global_return_list, 49)  # 平滑窗口调大一点
        plt.plot(mv_return, color='red', linewidth=2, label='Moving Avg')
    plt.title("PPO Training Curve (Random Maps)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MODEL_DIR, 'training_curve.png'))
    plt.show()
    print("Done.")