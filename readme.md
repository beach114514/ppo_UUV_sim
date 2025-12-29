# 基于PPO的UUV仿真项目 (UUV-PPO-Sim)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

**[基于PPO的UUV仿真]** 旨在记录自己的科研生活，探索高航速无人水下航行器（UUV）在复杂未知环境中的自主避障与导航。

## ✨ 功能特性

- 🚀 **快速验证**：支持加载已训练好的模型参数（`.pth` 文件），实现即时评估。
- 📦 **开箱即用**：通过运行 `PPO_UUVTest.py` 可迅速直观地查看智能体在随机障碍物环境中的避障表现。
- 🧠 **物理感知**：模型深入学习了 UUV 的动力学特性，包括高速下的“漂移”与“侧滑”效应。

## 🛠️ 安装说明

确保你的环境中已经安装了 **Conda / Python / CUDA**。

```bash
# 克隆仓库
git clone [https://github.com/beach114514/ppo_UUV_sim](https://github.com/beach114514/ppo_UUV_sim)

# 进入目录
cd ppo_UUV_sim

# 创建环境并同步依赖
conda env create -f environment.yml
conda env update --file environment.yml --prune

# 启动训练 (Windows PowerShell 示例)
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python .\PPO_UUVTrain.py
```
## ⚙️ 参数调整 (Configuration Guide)

项目的运行逻辑由模型加载标志位和环境配置字典共同决定：

### 1. 模型加载逻辑
- **`PTH_LOADED`**: 这是一个布尔标志位，决定智能体如何初始化 。
  - `True`: 系统将从指定路径加载预训练的 `.pth` 权重文件，适用于模型微调或直接进行性能测试 。
  - `False`: 智能体将从随机初始化状态开始学习，适用于全新的训练任务 。

### 2. 环境参数配置 (`MAP_CONFIG`)
这些参数直接定义了 UUV 运行的“无图环境”的物理约束和复杂程度：



| 参数 (Parameter) | 作用与影响 (Function and Impact) |
| :--- | :--- |
| **`map_size`** | 定义仿真区域的边界尺寸（例如 $700.0 \times 700.0$ 单位） 。 |
| **`n_obs`** | 生成障碍物的数量。增加此值会提高环境的“危险密度” 。 |
| **`n_goals`** | UUV 需要依次到达的目标点数量 。 |
| **`obs_radius_range`** | 设置障碍物的半径范围。较大的半径会增加寻找安全通路的难度 。 |
| **`map_refresh_freq`** | 控制障碍物布局刷新的频率。设为 `9999` 可有效锁定环境，确保评估的一致性 。 |
| **`fixed_map_seed`** | 固定随机种子，确保每次生成的障碍物位置完全一致，便于算法基准测试 。 |

### 3. 模式对比：训练 vs. 测试
虽然两种模式都使用物理感知 DRL 框架，但其操作目标不同：

#### 🟢 训练模式 (Training Mode)
- **核心目标**：通过 PPO 算法和协作奖励系统优化导航策略 。
- **运行机制**：智能体在动作中加入噪声进行**探索 (Exploration)**，以发现高奖励策略 。
- **动态特性**：在处理随机障碍物簇时，会经历“局部遗忘 (Partial Forgetting)”现象，这是策略泛化的关键阶段 。

#### 🔵 测试模式 (Testing Mode / Evaluation)
- **核心目标**：验证学到的策略在未知环境下的鲁棒性与安全性 。
- **运行机制**：采用**确定性策略 (Deterministic)**；智能体始终选择概率最高的动作，不含探索噪声。
- **关键指标**：重点评估成功率、高航速保持能力（如 $u \approx 8$ m/s）以及对 $2$ m 碰撞极限等安全边际的遵守情况 。


