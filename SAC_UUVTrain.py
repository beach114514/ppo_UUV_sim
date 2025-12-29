import random
import gymnasium
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
# 假设 UUV_Env.py 文件中包含了 UUV_MultiGoal_Env, custom_multi_goal_reward 和宏定义
from UUV_Env import UUV_MultiGoal_Env, custom_multi_goal_reward, \
    MAX_RUDDER_ANGLE, MAX_MOTOR_SPEED, SIM_BOUND, DT, R_COLLISION, R_FINAL_GOAL, R_STAGE_GOAL, R_ENERGY_PENALTY
import math
import rl_utils
import collections
from typing import Dict, List, Any
import gymnasium.vector  # <-- 【新增】用于并行化
import multiprocessing  # <-- 【新增】用于并行化

RL_MODE = 'trainmode'
alphainit = 0.05

# --- 并行化配置 ---
NUM_ENVS = 16  # <-- 【核心修改】设置并行环境数量 (利用 i5-13600KF 的多核性能)
reward_bili = 320000
reward_set = 0
# -----------------

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, MAX_MOTOR_SPEED, MAX_RUDDER_ANGLE):
        super(PolicyNetContinuous, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)

        self.mu = torch.nn.Linear(hidden_dim, action_dim)
        self.std = torch.nn.Linear(hidden_dim, action_dim)

        self.action_max_limit = torch.tensor([MAX_MOTOR_SPEED, MAX_RUDDER_ANGLE], dtype=torch.float)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        mu_vector = self.mu(x)  # [Batch_Size, 2]
        std_vector = F.softplus(self.std(x)) + 1e-6
        action_dist = Normal(mu_vector, std_vector)
        action_sample = action_dist.rsample()  # 采样值，形状 [Batch_Size, 2]

        action_tanh = torch.tanh(action_sample)
        action_tanh = action_tanh * 0.8  #限制运动

        action_max_limit_device = self.action_max_limit.to(x.device)
        action = action_tanh * action_max_limit_device  # [Batch_Size, 2]

        action_log_prob = action_dist.log_prob(action_sample)  # [Batch_Size, 2]
        tanh_correction = torch.log(1 - action_tanh.pow(2) + 1e-6)
        action_log_prob = action_log_prob - tanh_correction

        action_log_prob = action_log_prob.sum(dim=-1, keepdim=True)

        return action, action_log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, MAX_MOTOR_SPEED, MAX_RUDDER_ANGLE,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         MAX_MOTOR_SPEED, MAX_RUDDER_ANGLE).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(alphainit), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim

    # 【修改后的 take_action，处理批量状态输入】
    def take_action(self, state_batch: np.ndarray):
        """
        接收一个状态批次 (Batch of states)，形状: [NUM_ENVS, state_dim]
        返回一个动作批次 (Batch of actions)，形状: [NUM_ENVS, action_dim]
        """
        # 不再添加 [None]，直接使用传入的 batch
        state_tensor = torch.tensor(state_batch, dtype=torch.float).to(self.device)

        # self.actor 返回 (action_batch, log_prob)
        action_batch, _ = self.actor(state_tensor)

        # 使用 .numpy() 返回 NumPy 数组
        return action_batch.cpu().detach().numpy()

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, self.action_dim).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 【统一奖励缩放 (R + 15000) / 10000】
        # 这里使用统一的 (R + C) / K 确保稳定性
        rewards = (rewards) / reward_bili + reward_set

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    # (save_model 和 load_model 方法保持不变)
    def save_model(self, path):
        """保存 Actor 和两个 Critic 网络的参数到指定路径"""
        # 1. 创建目标目录 (如果不存在)
        import os
        if not os.path.exists(path):
            os.makedirs(path)

        # 2. 保存 Actor 的状态字典
        actor_path = os.path.join(path, 'actor.pth')
        torch.save(self.actor.state_dict(), actor_path)

        # 3. 保存 Critic 1 的状态字典
        critic1_path = os.path.join(path, 'critic_1.pth')
        torch.save(self.critic_1.state_dict(), critic1_path)

        # 4. 保存 log_alpha (可选)
        alpha_path = os.path.join(path, 'log_alpha.pth')
        torch.save(self.log_alpha, alpha_path)

        print(f"\n✅ Model parameters saved to: {path}")

    def load_model(self, path):
        # ... (此处省略 load_model 的代码，确保它也存在于类中) ...
        # 注意：你需要将 load_model 方法也粘贴回来，因为评估模式需要它。
        import os
        # 1. 加载 Actor
        actor_path = os.path.join(path, 'actor.pth')
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            print(f"Loaded Actor from {actor_path}")
        else:
            print(f"Actor file not found at {actor_path}")

        # 2. 加载 Critic 1 (如果需要继续训练)
        critic1_path = os.path.join(path, 'critic_1.pth')
        if os.path.exists(critic1_path):
            self.critic_1.load_state_dict(torch.load(critic1_path, map_location=self.device))
            # 目标 Critic 也要同步更新
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            print(f"Loaded Critic 1 from {critic1_path}")

        # 3. 加载 log_alpha (如果需要继续训练)
        alpha_path = os.path.join(path, 'log_alpha.pth')
        if os.path.exists(alpha_path):
            # 直接加载 tensor 并设置 requires_grad=True
            self.log_alpha = torch.load(alpha_path, map_location='cpu')
            self.log_alpha.requires_grad = True
            # 将优化器重新指向新的 log_alpha tensor
            self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                        lr=self.log_alpha_optimizer.param_groups[0]['lr'])
            print(f"Loaded log_alpha from {alpha_path}")


class ReplayBuffer:
    # (ReplayBuffer 类保持不变)
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1),
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32).reshape(-1, 1)
        }



    def size(self) -> int:
        return len(self.buffer)

if __name__ == '__main__':
    # --- 环境初始化 ---
    goal_positions = [(300, 100), (-100, 200), (0, -400)]
    obstacles = []


    #
    def make_env(rank):
        def _init():
            env = UUV_MultiGoal_Env(goal_positions, custom_multi_goal_reward, obstacles, 'none', max_steps=700)
            # 为每个环境设置不同的随机种子
            env.reset(seed=rank + 1)
            return env

        return _init


    # 【创建并行环境】使用 AsyncVectorEnv 可以在不同进程中并行执行 Env.step()
    uuv_env = gymnasium.vector.AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)])

    # --- 状态和动作维度来自单个环境的规格 ---
    state_dim = uuv_env.single_observation_space.shape[0]
    action_dim = uuv_env.single_action_space.shape[0]

    # 设置随机种子
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    actor_lr = 1e-4
    critic_lr = 1e-5
    alpha_lr = 3e-4
    return_list = []

    num_episodes = 5000
    hidden_dim = 256
    gamma = 0.99
    tau = 0.01  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 4096  # 保持大批量
    replay_buffer = ReplayBuffer(buffer_size)
    target_entropy = -10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    agent = SACContinuous(state_dim, hidden_dim, action_dim, MAX_MOTOR_SPEED, MAX_RUDDER_ANGLE,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)
    print(f"Agent initialized on {device}. Using {NUM_ENVS} parallel environments.")

    if RL_MODE == 'trainmode':

        # 【初始化并行环境状态】 states shape: [NUM_ENVS, state_dim]
        states, infos = uuv_env.reset()

        # 用于跟踪每个环境的回报 (未缩放，用于日志显示)
        episode_returns = np.zeros(NUM_ENVS)

        total_episodes_completed = 0
        total_episodes_target = int(num_episodes)
        train_progress = 0
        for i in range(60):
            # 这里的 tqdm total 应该是总回合数 / 10，我们使用一个大循环来适应并行结构
            with tqdm(total=total_episodes_target / 10, desc=f'Iteration {i}', initial=total_episodes_completed) as pbar:
                reachgoal_times = 0
                while total_episodes_completed < total_episodes_target / 10 * (i + 1):

                    # 1. Agent 批量决策: states shape [NUM_ENVS, state_dim] -> actions shape [NUM_ENVS, action_dim]
                    actions = agent.take_action(states)

                    # 2. 环境批量执行 Step
                    next_states, rewards, terminateds, truncateds, infos = uuv_env.step(actions)

                    num_terminated = np.sum(terminateds)

                    # 累加到统计变量中
                    reachgoal_times += num_terminated

                    dones = np.logical_or(terminateds, truncateds)

                    # 3. 存储和经验回放 (核心循环)
                    for env_idx in range(NUM_ENVS):
                        # 存储单个环境的过渡经验
                        replay_buffer.push(
                            states[env_idx],
                            actions[env_idx],
                            rewards[env_idx],  # 存储原始奖励
                            next_states[env_idx],
                            dones[env_idx]
                        )

                        # 累积当前环境的缩放回报 (用于日志显示)
                        # 你的缩放公式是 (R + 15000) / 10000
                        episode_returns[env_idx] += (rewards[env_idx]) / reward_bili +  + reward_set

                        # 4. 检查是否结束，进行重置和进度更新
                        if dones[env_idx]:
                            # 回合结束，记录最终回报
                            return_list.append(episode_returns[env_idx])

                            # 重置当前环境的回报计数器
                            episode_returns[env_idx] = 0

                            total_episodes_completed += 1
                            pbar.update(1)  # 更新 tqdm 进度条

                            if total_episodes_completed % 10 == 0:
                                pbar.set_postfix({
                                    'total_episodes':
                                        f'{total_episodes_completed}',
                                    'return':
                                        '%.3f' % np.mean(return_list[-10:]),
                                    'reach_goal':
                                        f'{reachgoal_times}'
                                })

                    # 5. 状态更新和训练步骤
                    states = next_states  # 下一时刻状态成为当前状态

                    # 更新 Agent (每收集 NUM_ENVS 个 step 就更新一次)
                    if replay_buffer.size() >= minimal_size:
                        train_batch = replay_buffer.sample(batch_size)
                        agent.update(train_batch)
            train_progress += 1
            if train_progress % 5 == 0:
                MODEL_SAVE_PATH = f'./model_save/1/{train_progress}'  # 定义保存路径
                agent.save_model(MODEL_SAVE_PATH)

        uuv_env.close()
        print("Parallel training environment closed.")

        print("Starting evaluation with rendering (single environment)...")


        single_env = UUV_MultiGoal_Env(goal_positions, custom_multi_goal_reward, obstacles, 'human', max_steps=500)

        state, _ = single_env.reset(seed=42)  # 使用一个固定的种子进行可重复评估
        done = False
        total_original_reward = 0
        step_count = 0
        for i in range(10):
            state, _ = single_env.reset(seed=42 + i)  # 增加种子偏移，确保评估多样性
            done = False
            # 3. 运行单个回合并渲染
            while not done:
                # 渲染当前帧 (需要 Matplotlib 窗口)
                single_env.render()

                # Agent 获取动作 (由于 take_action 接受 Batch，所以需要创建 [1, state_dim] 的批次)
                state_batch = np.array([state], dtype=np.float32)
                action_batch = agent.take_action(state_batch)  # 使用已训练的 agent
                action = action_batch[0]

                next_state, reward, terminated, truncated, info = single_env.step(action)
                done = terminated or truncated

                state = next_state
                total_original_reward += reward
                step_count += 1

                if step_count > 1000:  # 避免无限循环
                    break

        single_env.close()
        print(f"Evaluation finished. Total steps: {step_count}, Total Original Reward: {total_original_reward}")

        # 可视化代码
        mv_return = rl_utils.moving_average(return_list, 9)
        episodes_list = list(range(len(return_list)))

        plt.figure(figsize=(12, 6))
        # 1. 绘制原始回报
        plt.plot(episodes_list, return_list, label='Episode Return (Raw)', alpha=0.5)

        # 2. 绘制移动平均回报 (确保索引对齐)
        # 移动平均长度比原始列表短 9-1=8 个点，但这在长曲线上可以忽略
        plt.plot(episodes_list, mv_return, label='Moving Average (Window 9)', color='red', linewidth=2)

        plt.xlabel('Episodes')
        plt.ylabel('Scaled Returns')
        plt.title('SAC Training on UUV Multi-Goal Environment')
        plt.legend()
        plt.grid(True)
        plt.show()