import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

# ====================================================
# I. 宏定义与经验系数
# ====================================================
DT = 0.1  # [s] 运动周期 (Time Step)
MAX_RUDDER_ANGLE = math.pi / 6  # [rad] 最大舵角 (30度)
MAX_MOTOR_SPEED = 70.0  # [RPM] 电机最大转速
SIM_BOUND = 1000.0  # [m] 仿真区域边界 (-50m to 50m)

GOAL_THRESHOLD = 3  # [m] 到达目标的距离阈值
SENSING_RANGE = 200  # 声纳探测距离

# ====================================================
# II. 奖励函数接口
# ====================================================

# --- 奖励函数超参数 ---
R_COLLISION = -1500.0       # 撞墙/碰撞惩罚 (大额负分，极力避免)
R_FINAL_GOAL = 1000.0      # 抵达最终终点奖励 (最大的诱惑)
R_STAGE_GOAL = 400.0       # 抵达途经点(子目标)奖励 (阶段性激励)
R_DIST_SCALE = 1.0         # 距离引导奖励系数 (越靠近目标分越高，Dense Reward)
R_TIME_PENALTY = -0.5     # 每步时间惩罚 (逼迫走捷径，越快越好)
R_HEADING = 0.0            # 航向误差惩罚 (当前设为0，未启用)
R_ACTION_SMOOTH = 0.05     # 动作平滑惩罚 (防止舵角/油门剧烈抖动)
R_YAW_DAMPING = 0.01       # 角速度阻尼惩罚 (防止自旋过快)
R_RUDDER_PENALTY = 0.005   # 舵角使用惩罚 (鼓励少打舵，节能)
R_THROTTLE_PENALTY = 0.00  # 油门能耗惩罚 (当前设为0，未启用)
R_COURSE_ALIGNMENT = 1   # 航向对齐奖励 (鼓励船头一直对准目标)
APPROACH_RADIUS = 30.0     # 进近半径 (进入此范围可能触发减速逻辑)

REWARD_SCALE = 100.0       # 奖励归一化缩放 (用于 PPO 训练稳定数值)

UUV_M = 100.0              # 物理质量 (kg)
MU = 1.05 * UUV_M          # 纵向(Surge)附加质量 (包含水的惯性)
MV = 1.5 * UUV_M           # 横向(Sway)附加质量 (横向推水更重)
IZ = 150                   # 偏航转动惯量 (转弯的难易程度)
DU = 2.0                   # 纵向线性阻尼 (水的直行阻力)
DV = 80.0                  # 横向线性阻尼 (水的横移阻力)
DR = 30.0                   # 偏航角阻尼 (水的旋转阻力)
KN = 200.0 / (MAX_MOTOR_SPEED ** 2)  # 螺旋桨推力系数 (转速转推力的比例)
KRUDDER = 2.0             # 舵升力系数 (舵角转转向力的比例)
L_rudder = 1.2             # 舵力臂长度 (舵距离重心的距离，越长转向力矩越大)


def custom_multi_goal_reward(env, old_eta, new_eta, nu, action, last_action,
                             is_goal_achieved: bool, terminated: bool, truncated: bool) -> float:
    # 1. 撞墙判定
    if terminated and not is_goal_achieved:
        return R_COLLISION

    reward = 0.0
    uuv_pos = new_eta[:2]

    # 定义局部常量
    UUV_BODY_RADIUS = 2.0
    NORM_DIST = 0.70  # 警戒范围系数 (30%)
    DANGER_DIST = 0.20  # 危险范围系数 (15%)

    # ====================================================
    # A. 斥力计算 (Max 逻辑 - 最终修正版)
    # ====================================================

    # 1. 初始化变量
    max_threat_force = 0.0  # 记录这一刻面临的最大单一威胁
    total_speed_penalty = 0.0  # 超速惩罚累加

    for obs in env.obstacles:
        center = obs['center']
        r = obs['radius']

        dist_c = np.linalg.norm(uuv_pos - center)
        dist_calc = max(0.0, dist_c - r - UUV_BODY_RADIUS)

        # 动态阈值
        rep_thresh = max(r * NORM_DIST, 10.0)
        dang_thresh = max(r * DANGER_DIST, 5.0)

        # 只有进入警戒圈才计算
        if dist_calc < rep_thresh:
            # --- 1. 计算基础柔性斥力 (外层) ---
            norm_dist = dist_calc / rep_thresh
            single_force = 4.0 * ((1.0 - norm_dist) ** 3)

            # --- 2. 叠加刚性斥力 (内层) ---
            if dist_calc < dang_thresh:
                # [Fix Start] --------------------------------------------
                # 1. 获取当前航向角
                psi = new_eta[2]

                # 2. 将体坐标系速度 nu[u, v] 转换到 世界坐标系 [vx, vy]
                # 旋转矩阵公式
                w_vx = nu[0] * np.cos(psi) - nu[1] * np.sin(psi)
                w_vy = nu[0] * np.sin(psi) + nu[1] * np.cos(psi)
                world_vel = np.array([w_vx, w_vy])

                # 3. 计算指向向量 (世界坐标系)
                dir_to_obs = (center - uuv_pos) / (dist_c + 1e-6)

                # 4. 现在两个向量都在世界坐标系，点积才成立
                speed_towards = np.dot(world_vel, dir_to_obs)
                # [Fix End] ----------------------------------------------

                limit_speed = (dist_calc / dang_thresh) * 0.45 + 0.1

                # A. 超速惩罚
                if speed_towards > limit_speed:
                    excess_speed = speed_towards - limit_speed
                    total_speed_penalty += excess_speed * 20.0

                # B. 危险区势场叠加
                norm_dang = dist_calc / dang_thresh
                single_force += 30.0 * ((1.0 - norm_dang) ** 2)

                # C. 绝对禁区 (2m以内)
                if dist_calc < 2.0:
                    single_force += 150.0

            # --- 取最大值逻辑 ---
            # 更新当前最大的威胁
            if single_force > max_threat_force:
                max_threat_force = single_force

    # 3. 最终扣分 (循环结束后执行一次)
    reward -= min(max_threat_force, 300.0)
    reward -= total_speed_penalty

    # ====================================================
    # B. 距离与航向奖励 (保持不变)
    # ====================================================
    current_goal_pos = env.all_goals[env.current_goal_idx]
    cur_dist_goal = np.linalg.norm(uuv_pos - current_goal_pos)
    old_dist_goal = np.linalg.norm(old_eta[:2] - current_goal_pos)

    reward += (old_dist_goal - cur_dist_goal) * R_DIST_SCALE

    vec_move = new_eta[:2] - old_eta[:2]
    vec_goal = current_goal_pos - old_eta[:2]
    move_len = np.linalg.norm(vec_move)
    goal_len = np.linalg.norm(vec_goal)

    if move_len > 1e-4 and goal_len > 1e-4:
        cosine = np.dot(vec_move, vec_goal) / (move_len * goal_len)
        reward += cosine * R_COURSE_ALIGNMENT
    # ====================================================

    # C. 动作与状态约束

    # ====================================================
    # if last_action is not None:
    #     delta_action = action - last_action
    #     r_smooth = np.sum(delta_action ** 2)
    #     reward -= r_smooth * R_ACTION_SMOOTH
    #
    # r_yaw = nu[2]
    # reward -= (r_yaw ** 2) * R_YAW_DAMPING
    # reward -= (action[1] ** 2) * R_RUDDER_PENALTY
    # ====================================================
    # D. 生存与事件 (保持不变)
    # ====================================================
    reward += R_TIME_PENALTY

    if is_goal_achieved:
        reward += R_STAGE_GOAL
        if terminated:
            reward += R_FINAL_GOAL

    if truncated:
        reward -= 500.0

    return reward


# ====================================================
# III. 地图生成器
# ====================================================
def generate_valid_map(
        map_size=400.0,
        n_obstacles=4,
        n_goals=3,
        obs_radius_range=(30, 60),
        safe_start_radius=50.0,
        min_gap=40.0,  # 稍微改小一点，允许障碍物紧凑一些
        seed=None
):
    # 使用局部随机生成器
    rng = np.random.RandomState(seed)

    obstacles = []
    goals = []

    # --- 1. 先生成目标点 (Goals First) ---
    # 我们假设起点总是 (0,0)，这是 UUV reset 时的默认位置
    current_pos = np.array([0.0, 0.0])

    # 临时列表，包含起点和所有目标点，用于构建“路径线段”
    path_points = [current_pos]

    attempts = 0
    while len(goals) < n_goals and attempts < 2000:
        attempts += 1
        # 目标点依然全图随机，保持长距离导航的需求
        pos = rng.uniform(-map_size * 0.8, map_size * 0.8, size=2)

        # 检查新目标点离上一个点的距离，不能太近
        last_pos = path_points[-1]
        if np.linalg.norm(pos - last_pos) < 400.0:  # 保证每段路至少150m长
            continue

        # 检查目标点之间不能重叠太近
        conflict = False
        for g in goals:
            if np.linalg.norm(pos - g) < 150.0:
                conflict = True
                break
        if conflict: continue

        goals.append(pos)
        path_points.append(pos)

    # --- 2. 在路径上针对性生成障碍物 (Obstacles on Path) ---
    # 定义冲突检测函数
    def is_conflict(pos, radius, existing_obstacles, check_start=True):
        # 检查是否盖住了起点
        if check_start:
            dist_to_start = np.linalg.norm(pos)
            if dist_to_start < (radius + safe_start_radius):
                return True
        # 检查是否盖住了任意目标点
        for g in goals:
            if np.linalg.norm(pos - g) < (radius + 20.0):  # 给目标点留20m空隙
                return True
        # 检查是否和其他障碍物重叠
        for obs in existing_obstacles:
            dist = np.linalg.norm(pos - obs['center'])
            if dist < (radius + obs['radius'] + min_gap):
                return True
        return False

    attempts = 0
    while len(obstacles) < n_obstacles and attempts < 2000:
        attempts += 1

        # A. 随机选一段路径 (例如 Start->Goal1 或 Goal1->Goal2)
        segment_idx = rng.randint(0, len(path_points) - 1)
        p_start = path_points[segment_idx]
        p_end = path_points[segment_idx + 1]

        # B. 在线段上插值
        # t 控制在 0.2 到 0.8 之间，不要堵在门口也不要堵在终点脸上
        t = rng.uniform(0.2, 0.8)

        # 线性插值公式
        vec_path = p_end - p_start
        base_pos = p_start + t * vec_path

        # C. 增加垂直扰动 (Jitter)
        vec_perp = np.array([-vec_path[1], vec_path[0]])
        # 归一化
        norm = np.linalg.norm(vec_perp)
        if norm > 1e-6:
            vec_perp /= norm

        # 偏移量：在路径左右摇摆 (-50m 到 50m)
        jitter = rng.uniform(-50.0, 50.0)
        pos = base_pos + jitter * vec_perp

        # D. 随机半径
        r = rng.uniform(obs_radius_range[0], obs_radius_range[1])

        # E. 检查冲突
        if not is_conflict(pos, r, obstacles, check_start=True):
            obstacles.append({'center': pos, 'radius': r})

    fallback_attempts = 0
    while len(obstacles) < n_obstacles and fallback_attempts < 1000:
        fallback_attempts += 1
        r = rng.uniform(obs_radius_range[0], obs_radius_range[1])
        pos = rng.uniform(-map_size * 0.9, map_size * 0.9, size=2)
        if not is_conflict(pos, r, obstacles, check_start=True):
            obstacles.append({'center': pos, 'radius': r})

    return obstacles, goals


# ====================================================
# IV. 环境类定义
# ====================================================
class UUV_MultiGoal_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 reward_fn: Callable,
                 goal_positions: List[Tuple[float, float]] = None,
                 obstacles: List[Tuple[float, float, float]] = None,
                 map_config: dict = None,
                 render_mode='human',
                 max_steps=5000):

        super(UUV_MultiGoal_Env, self).__init__()

        self.reward_fn = reward_fn

        # 默认地图配置
        if map_config is None:
            self.map_config = {
                'map_size': 500.0,
                'n_obs': 6,
                'n_goals': 3,
                'obs_radius_range': (25, 45),
                'map_refresh_freq': 80
            }
        else:
            self.map_config = map_config

        self.global_episode_count = 0
        self.cached_obstacles = []
        self.cached_goals = []
        self.max_steps = max_steps
        self.render_mode = render_mode

        # --- 初始固定目标与障碍物处理 ---
        self.all_goals = []
        if goal_positions is not None:
            self.all_goals = [np.array(g, dtype=np.float32) for g in goal_positions]
            import copy
            self.cached_goals = copy.deepcopy(self.all_goals)

        self.obstacles = []
        if obstacles is not None:
            for obs in obstacles:
                self.obstacles.append({
                    'center': np.array(obs[:2], dtype=np.float32),
                    'radius': obs[2]
                })
            import copy
            self.cached_obstacles = copy.deepcopy(self.obstacles)

        self.current_goal_idx = 0
        self.goal_patches = []
        self.ray_lines = []
        self.fig, self.ax = None, None

        self.SENSING_RANGE = SENSING_RANGE
        self.SAFE_DISTANCE_THRESHOLD = 0.5

        # 动作与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.current_eta = np.zeros(3, dtype=np.float32)
        self.current_nu = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0

        self.n_rays = 19
        self.sensor_span = np.pi
        self.ray_angles = np.linspace(-self.sensor_span / 2, self.sensor_span / 2, self.n_rays)
        total_obs_dim = 5 + self.n_rays
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32)

    def _cast_ray(self, uuv_pos, uuv_angle, ray_rel_angle):
        global_angle = uuv_angle + ray_rel_angle
        ray_dir = np.array([np.cos(global_angle), np.sin(global_angle)])
        min_hit_dist = self.SENSING_RANGE
        for obs in self.obstacles:
            vec_to_circle = obs['center'] - uuv_pos
            projection = np.dot(vec_to_circle, ray_dir)
            if projection < 0: continue
            dist_sq = np.linalg.norm(vec_to_circle) ** 2
            perp_dist_sq = dist_sq - projection ** 2
            if perp_dist_sq > obs['radius'] ** 2: continue
            hit_dist = projection - np.sqrt(obs['radius'] ** 2 - perp_dist_sq)
            if 0 < hit_dist < min_hit_dist:
                min_hit_dist = hit_dist
        return min_hit_dist

    def _kinematics_step(self, eta, nu):
        psi = eta[2]
        J = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]], dtype=np.float32)
        dot_eta = J @ nu
        next_eta = eta + dot_eta * DT
        next_eta[2] = (next_eta[2] + np.pi) % (2 * np.pi) - np.pi
        return next_eta

    def _get_obs(self):
        x, y, psi = self.current_eta
        current_goal_pos = self.all_goals[self.current_goal_idx]
        distance = np.linalg.norm(current_goal_pos - np.array([x, y]))
        angle_to_goal = np.arctan2(current_goal_pos[1] - y, current_goal_pos[0] - x)
        angle_error = (angle_to_goal - psi + np.pi) % (2 * np.pi) - np.pi

        norm_x = np.clip(x / SIM_BOUND, -1.0, 1.0)
        norm_y = np.clip(y / SIM_BOUND, -1.0, 1.0)
        norm_psi = psi / np.pi
        norm_dist = np.clip(distance / (SIM_BOUND * 2), 0.0, 1.0)
        norm_angle_err = angle_error / np.pi

        nav_obs = np.array([norm_x, norm_y, norm_psi, norm_dist, norm_angle_err], dtype=np.float32)
        ray_readings = [np.clip(self._cast_ray(self.current_eta[:2], psi, ang) / self.SENSING_RANGE, 0.0, 1.0) for ang
                        in self.ray_angles]
        return np.concatenate([nav_obs, np.array(ray_readings, dtype=np.float32)])

    def _get_info(self):
        return {"speed_u": self.current_nu[0], "current_goal_idx": self.current_goal_idx}

    def _check_termination_and_goal(self):
        terminated = False
        truncated = False
        is_goal_achieved = False

        min_dist = float('inf')
        if self.obstacles:
            obs_centers = np.array([o['center'] for o in self.obstacles])
            obs_radii = np.array([o['radius'] for o in self.obstacles])
            dists = np.linalg.norm(obs_centers - self.current_eta[:2], axis=1) - obs_radii
            min_dist = np.min(dists)

        if np.any(np.abs(self.current_eta[:2]) > SIM_BOUND):
            truncated = True

        if min_dist <= self.SAFE_DISTANCE_THRESHOLD:
            terminated = True

        dist_to_goal = np.linalg.norm(self.current_eta[:2] - self.all_goals[self.current_goal_idx])
        if dist_to_goal < GOAL_THRESHOLD:
            is_goal_achieved = True
            if self.current_goal_idx < len(self.all_goals) - 1:
                self.current_goal_idx += 1
            else:
                terminated = True

        return terminated, truncated, is_goal_achieved

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        is_first_episode = (self.global_episode_count == 0)
        has_initial_map = (len(self.cached_obstacles) > 0 or len(self.cached_goals) > 0)

        freq = self.map_config.get('map_refresh_freq', 320)
        fixed_seed = self.map_config.get('fixed_map_seed', None)  # 读取固定种子

        refresh_needed = (self.global_episode_count % freq == 0)

        # 1. 如果是第一局且 __init__ 传入了初始地图，优先使用 __init__ 的，不生成
        if is_first_episode and has_initial_map:
            pass

        # 2. 判断是否需要生成新地图
        # 逻辑 A (固定种子模式): 设定了固定种子，且当前没有缓存地图 -> 生成一次
        # 逻辑 B (自动刷新模式): 没设固定种子，且 (到了刷新周期 或 没缓存) -> 生成
        elif (fixed_seed is not None and not self.cached_obstacles) or \
                (fixed_seed is None and (refresh_needed or not self.cached_obstacles)):

            # === 计算种子 ===
            if fixed_seed is not None:
                shared_seed = fixed_seed
                # print(f"[Map Gen] Using FIXED Seed: {shared_seed}")
            else:
                map_cycle_id = self.global_episode_count // freq
                shared_seed = map_cycle_id + 13579
                # print(f"[Map Gen] Sync Map ID: {map_cycle_id} (Seed: {shared_seed})")

            # 生成并缓存
            self.cached_obstacles, self.cached_goals = generate_valid_map(
                map_size=self.map_config['map_size'],
                n_obstacles=self.map_config['n_obs'],
                n_goals=self.map_config['n_goals'],
                obs_radius_range=self.map_config['obs_radius_range'],
                seed=shared_seed
            )

        import copy
        self.obstacles = copy.deepcopy(self.cached_obstacles)
        self.all_goals = copy.deepcopy(self.cached_goals)

        self.current_step = 0
        self.current_goal_idx = 0
        self.current_eta = np.zeros(3, dtype=np.float32)
        self.current_eta[2] = self.np_random.uniform(-np.pi, np.pi)
        self.current_nu = np.zeros(3, dtype=np.float32)
        self.last_action = None
        self.path_history = []

        self.global_episode_count += 1

        if self.render_mode == 'human':
            self.close()
            self.render()

        return self._get_obs(), self._get_info()

    def step(self, action):
        if np.any(np.isnan(action)):
            return self._get_obs(), -500.0, True, False, {}

        normalized_motor, normalized_rudder = action
        motor_speed = (normalized_motor + 1) / 2 * MAX_MOTOR_SPEED
        rudder_angle = normalized_rudder * MAX_RUDDER_ANGLE

        u, v, r = self.current_nu
        F_push = KN * (motor_speed ** 2)
        F_rudder_lift = (motor_speed ** 2) * rudder_angle * KRUDDER

        u_dot = (F_push - DU * u * math.fabs(u) + MV * v * r) / MU
        v_dot = (F_rudder_lift - DV * v * math.fabs(v) - MU * u * r) / MV
        r_dot = (F_rudder_lift * L_rudder - DR * r * math.fabs(r) + (MU - MV) * u * v) / IZ

        nu = np.array([u + u_dot * DT, v + v_dot * DT, r + r_dot * DT], dtype=np.float32)
        nu = np.clip(nu, -2.0, [8.0, 2.0, 2.0])

        old_eta = self.current_eta.copy()
        self.current_eta = self._kinematics_step(old_eta, nu)
        self.current_nu = nu

        terminated, truncated, is_goal_achieved = self._check_termination_and_goal()
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True

        reward = self.reward_fn(self, old_eta, self.current_eta, nu, action, self.last_action,
                                is_goal_achieved, terminated, truncated)
        info = self._get_info()
        info['is_goal_achieved'] = is_goal_achieved

        if terminated:
            info['is_success'] = is_goal_achieved  # 只有通关时由 check 函数置为 True

        self.last_action = action.copy()

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_title("UUV Multi-Goal Planning")
                self.ax.set_xlim(-SIM_BOUND * 1.1, SIM_BOUND * 1.1)
                self.ax.set_ylim(-SIM_BOUND * 1.1, SIM_BOUND * 1.1)
                self.ax.axis('equal')
                self.ax.grid(True)

                self.ray_lines = [self.ax.plot([], [], 'r-', linewidth=0.5, alpha=0.3)[0] for _ in range(self.n_rays)]
                self.goal_patches = [self.ax.plot(g[0], g[1], 'go', markersize=10, label=f'Goal {i}')[0] for i, g in
                                     enumerate(self.all_goals)]

                for obs in self.obstacles:
                    self.ax.add_patch(plt.Circle(obs['center'], obs['radius'], color='gray', alpha=0.5))

                self.ship_shape = np.array([[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]])
                self.uuv_poly = plt.Polygon(self.ship_shape, color='blue', alpha=0.8, label='UUV')
                self.ax.add_patch(self.uuv_poly)
                self.path_line, = self.ax.plot([], [], 'b--', linewidth=1, alpha=0.5)

            x, y, psi = self.current_eta
            self.path_history.append((x, y))
            # if len(self.path_history) > 3000: self.path_history.pop(0)
            if self.path_history: self.path_line.set_data(*zip(*self.path_history))

            # ray_dists = [self._cast_ray(self.current_eta[:2], psi, ang) for ang in self.ray_angles]
            # for i, line in enumerate(self.ray_lines):
            #     angle = psi + self.ray_angles[i]
            #     line.set_data([x, x + ray_dists[i] * np.cos(angle)], [y, y + ray_dists[i] * np.sin(angle)])

            c, s = np.cos(psi), np.sin(psi)
            R = np.array(((c, -s), (s, c)))
            self.uuv_poly.set_xy((self.ship_shape * 2.0) @ R.T + np.array([x, y]))

            for i, patch in enumerate(self.goal_patches):
                patch.set_color('r' if i == self.current_goal_idx else 'g')

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# ====================================================
# V. 示例运行 (自定义地图测试)
# ====================================================
if __name__ == '__main__':
    # 1. 【自定义目标点】格式: [(x, y), (x, y), ...]
    # 这里的坐标是相对于起点 (0,0) 的绝对坐标
    my_custom_goals = [
        (300.0, 0.0),  # 第1个目标：正前方 300米
        (300.0, 300.0),  # 第2个目标：右上方
        (0.0, 0.0)  # 第3个目标：回到起点
    ]

    # 2. 【自定义障碍物】格式: [(x, y, radius), ...]
    # 比如在去往第一个目标的必经之路上放一个大圆
    my_custom_obstacles = [
        # (150.0, 0.0, 40.0),  # 在 X=150 处挡一个半径40的大圆
        # (150.0, 60.0, 20.0),  # 旁边再放个小圆，制造缝隙
        # (300.0, 150.0, 30.0),  # 在去往第二个目标的路上放一个
    ]

    # 3. 【配置地图参数】
    test_map_config = {
        'map_size': 500.0,
        'n_obs': 0,  # 自动生成数量设为0（不影响手动传入的）
        'n_goals': 0,
        'obs_radius_range': (20, 50),
    }

    # 4. 【初始化环境】
    env = UUV_MultiGoal_Env(
        reward_fn=custom_multi_goal_reward,  # 必须传入奖励函数
        goal_positions=my_custom_goals,  # <--- 传入自定义目标
        obstacles=my_custom_obstacles,  # <--- 传入自定义障碍物
        map_config=test_map_config,  # <--- 传入配置
        render_mode='human',  # 开启可视化
        max_steps=5000  # 单局最大步数
    )

    obs, _ = env.reset()
    print("=== 自定义地图测试开始 ===")
    print(f"Goals: {my_custom_goals}")
    print(f"Obstacles: {my_custom_obstacles}")

    try:
        for i in range(10000):
            angle_error = obs[4]
            dist_to_goal = obs[3]

            rudder_cmd = np.clip(angle_error * 8.0, -1.0, 1.0)

            motor_cmd = 0.8 if dist_to_goal > 0.05 else 0.4

            action = np.array([motor_cmd, rudder_cmd], dtype=np.float32)

            obs, r, term, trunc, info = env.step(action)

            if i % 20 == 0:
                print(f"Step {i}: Reward={r:.2f}, Goal_Idx={info['current_goal_idx']}")
                print(env.current_eta)

            if term or trunc:
                status = "Success" if info.get('is_goal_achieved', False) else "Fail"
                print(f"Episode Finished. Result: {status}")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("测试手动中断")
    finally:
        env.close()

        print("环境已关闭")
