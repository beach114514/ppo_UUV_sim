###### **ver2 ：增加物理模型真实性，以pid全速冲击会产生大范围的漂移，以下为参数**

&nbsp;   MAX\_STEPS = 1500，GOAL\_POSITIONS = \[(300, 100), (-100, 200), (0, -400)] ， OBSTACLES = \[] ，TOTAL\_EPISODES = 10000， NUM\_ITERATIONS = 250

&nbsp;   EPISODES\_PER\_ITER = TOTAL\_EPISODES // NUM\_ITERATIONS ，EPISODES\_PER\_LOG = 80，TOTAL\_ROUNDS = 200，STEPS\_PER\_UPDATE = 4096

&nbsp;   REWARD\_SCALE = 300.0，ACTOR\_LR = 3e-5，CRITIC\_LR = 3e-5，HIDDEN\_DIM = 256，GAMMA = 0.99， LAMBDA = 0.95，EPOCHS = 3，EPS\_CLIP = 0.1，ENT\_COEF = 0.01

R\_COLLISION      = -250.0   ，R\_FINAL\_GOAL     = 600.0   R\_STAGE\_GOAL     = 200.0  R\_DIST\_SCALE     = 0.2  R\_TIME\_PENALTY   = -0.01 R\_HEADING        = 0.0  R\_ENERGY\_PENALTY = 0.0

###### **ver21 ：修复了关于指向性的问题，在当前物理模型下uuv头部与航迹不一定相符的BUG，修改为航迹指向奖励**

参数同上

###### **stage1参数**

###### **完成第一阶段的循迹，能最小偏移到达三个目标点，目前需要继续优化轨迹，预计stage2优化轨迹**

