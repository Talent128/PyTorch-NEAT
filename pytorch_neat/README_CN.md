# PyTorch-NEAT 核心库说明文档

本文档详细说明 `pytorch_neat` 核心库中各个模块的功能和用法。

## 📚 目录

1. [核心模块](#核心模块)
2. [网络类型](#网络类型)
3. [环境相关](#环境相关)
4. [工具模块](#工具模块)
5. [使用示例](#使用示例)

---

## 核心模块

### 1. `multi_env_eval.py` - 多环境评估器

**功能**：批量评估神经网络在多个环境实例中的表现

**主要类**：
- `MultiEnvEvaluator`: 同时在多个环境中评估基因组，返回平均适应度

**特点**：
- ✅ 提高评估效率（批量处理）
- ✅ 提高评估稳定性（平均多次运行）
- ✅ 测试泛化能力（在不同环境配置下测试）

**使用方法**：
```python
from pytorch_neat.multi_env_eval import MultiEnvEvaluator

evaluator = MultiEnvEvaluator(
    make_net=make_net_function,      # 网络创建函数
    activate_net=activate_function,   # 网络激活函数
    make_env=make_env_function,       # 环境创建函数
    batch_size=4,                     # 批量大小
    max_env_steps=1000                # 最大步数
)

fitness = evaluator.eval_genome(genome, config)
```

---

### 2. `neat_reporter.py` - 训练日志记录器

**功能**：记录NEAT训练过程中的详细统计信息到JSON文件

**主要类**：
- `LogReporter`: 继承自NEAT的BaseReporter，记录每代的统计数据

**记录内容**：
- 代数（generation）
- 平均适应度和标准差（fitness_avg, fitness_std）
- 最佳适应度（fitness_best, fitness_best_val）
- 最佳网络结构（n_neurons_best, n_conns_best）
- 种群统计（pop_size, n_species, n_extinctions）
- 时间统计（time_elapsed, time_elapsed_avg）

**使用方法**：
```python
from pytorch_neat.neat_reporter import LogReporter

logger = LogReporter("training.log", eval_genome_function)
population.add_reporter(logger)
```

**日志格式**：每行一个JSON对象，便于后续分析和可视化

---

## 网络类型

### 🔍 网络类型对比表

| 特性 | RecurrentNet | AdaptiveLinearNet | AdaptiveNet |
|------|-------------|-------------------|-------------|
| **网络结构** | 输入→隐藏→输出 | 输入→输出（无隐藏层） | 输入→隐藏→输出 |
| **循环连接** | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| **自适应权重** | ❌ 固定权重 | ✅ 动态更新 | ✅ 动态更新 |
| **空间编码** | ❌ 无 | ✅ 使用坐标 | ✅ 使用坐标 |
| **CPPN生成** | ❌ 无 | ✅ 生成权重更新规则 | ✅ 生成权重更新规则 |
| **记忆机制** | 隐藏层状态 | 权重变化 | 隐藏层状态+权重变化 |
| **计算复杂度** | 中等 | 低 | 高 |
| **适用任务** | 一般控制任务 | 需要快速适应的任务 | 复杂的记忆和适应任务 |
| **训练速度** | 快 | 中等 | 慢 |
| **示例** | CartPole | T-Maze | 复杂导航任务 |

**选择建议**：
- 🎯 **简单控制任务**（如CartPole）→ 使用 `RecurrentNet`
- 🧠 **需要短期记忆**（如T-Maze）→ 使用 `AdaptiveLinearNet`
- 🚀 **复杂任务需要强大能力** → 使用 `AdaptiveNet`
- ⚡ **追求训练速度** → 优先使用 `RecurrentNet`
- 🎓 **研究自适应机制** → 使用自适应网络

---

### 3. `recurrent_net.py` - 递归神经网络

**功能**：实现支持循环连接的递归神经网络

**主要类**：
- `RecurrentNet`: 递归神经网络实现，支持隐藏层和输出层之间的循环连接

**特点**：
- 🔄 支持循环连接（时间维度上的记忆）
- 📊 使用稀疏矩阵表示连接（节省内存）
- ⚡ PyTorch实现，支持批量处理
- 🎛️ 可配置激活函数、内部迭代步数等

**网络结构**：
```
输入层 → 隐藏层 ↺ → 输出层 ↺
         ↓           ↓
      (循环)      (循环)
```

**关键参数**：
- `n_internal_steps`: 内部循环迭代次数（增加可提升计算能力）
- `use_current_activs`: 是否使用当前时刻的隐藏层激活值
- `dtype`: 数据类型（默认torch.float64）

**使用方法**：
```python
from pytorch_neat.recurrent_net import RecurrentNet

net = RecurrentNet.create(
    genome,                    # NEAT基因组
    config,                    # NEAT配置
    batch_size=1,              # 批量大小
    activation=sigmoid_activation,  # 激活函数
    n_internal_steps=1         # 内部迭代次数
)

outputs = net.activate(inputs)  # 前向传播
net.reset(batch_size)           # 重置网络状态
```

---

### 4. `adaptive_linear_net.py` - 自适应线性网络

**功能**：实现权重可以根据神经元激活动态调整的自适应网络

**主要类**：
- `AdaptiveLinearNet`: 自适应权重的线性网络，权重通过CPPN生成

**核心概念**：
- **自适应权重**：权重在每个时间步根据输入、输出和当前权重动态更新
- **空间编码**：使用2D坐标表示神经元位置
- **CPPN生成**：使用组合模式生成网络（CPPN）来生成权重更新规则

**权重更新公式**：
```
Δw = CPPN(x_in, y_in, x_out, y_out, pre, post, w)
w_new = w_old + Δw (如果连接存在)
```

**适用场景**：
- 需要短期记忆的任务（T-Maze等）
- 需要在线学习的任务
- 需要快速适应的任务

**使用方法**：
```python
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.activations import tanh_activation

# 定义神经元空间坐标
input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]

net = AdaptiveLinearNet.create(
    genome,
    config,
    input_coords=input_coords,
    output_coords=output_coords,
    weight_threshold=0.4,       # 权重阈值
    activation=tanh_activation,
    batch_size=4
)

outputs = net.activate(inputs)  # 前向传播并更新权重
net.reset()                     # 重置网络状态和权重
```

---

### 5. `adaptive_net.py` - 自适应递归网络

**功能**：结合递归结构和自适应权重的高级网络

**区别**：
- `adaptive_linear_net.py`：只有输入→输出的简单连接
- `adaptive_net.py`：包含隐藏层和循环连接的完整网络

---

### 6. `cppn.py` - 组合模式生成网络

**功能**：实现CPPN（Compositional Pattern Producing Networks）

**主要功能**：
- `create_cppn()`: 从NEAT基因组创建CPPN网络
- `clamp_weights_()`: 对权重进行阈值处理和限幅
- `get_coord_inputs()`: 生成神经元坐标输入

**CPPN用途**：
- 生成权重矩阵（HyperNEAT）
- 生成权重更新规则（自适应网络）
- 利用空间结构信息

**CPPN输入**：
- `x_in, y_in`: 输入神经元坐标
- `x_out, y_out`: 输出神经元坐标
- `pre, post`: 神经元激活值
- `w`: 当前权重值

---

## 环境相关

### 7. `t_maze.py` - T型迷宫环境

**功能**：实现需要记忆能力的T型迷宫导航任务

**环境说明**：
```
    L     R      ← 目标位置（奖励）
    |-----|
        |
        |         ← 走廊
        |
        S         ← 起点（显示提示色）
```

**任务流程**：
1. 在起点显示颜色提示（指示高奖励在左还是右）
2. 智能体向前走过走廊（此时看不到颜色）
3. 到达T字路口，需要根据记住的颜色选择方向
4. 选对方向获得高奖励，选错方向获得低奖励

**关键参数**：
- `hall_len`: 走廊长度（越长越难，需要更好的记忆）
- `n_trials`: 试验次数
- `init_reward_side`: 初始高奖励方向（0=左，1=右）
- `reward_flip_mean`: 奖励方向切换的平均试验数

**观测空间**：4维
- [0]: 左侧是否有墙
- [1]: 前方是否有墙
- [2]: 右侧是否有墙
- [3]: 当前颜色（奖励提示）

**动作空间**：3个离散动作
- 0: 向左移动
- 1: 向前移动
- 2: 向右移动

---

### 8. `strict_t_maze.py` - 严格T型迷宫

**功能**：更严格版本的T型迷宫，要求更精确的控制

**与t_maze.py的区别**：
- 添加了转向限制
- 更严格的墙壁碰撞惩罚
- 需要更精细的策略

---

### 9. `turning_t_maze.py` - 转向T型迷宫

**功能**：需要显式转向操作的T型迷宫变种

**特点**：
- 智能体有朝向概念
- 需要转向动作（左转、右转）
- 更接近真实机器人导航

---

### 10. `maze.py` - 通用迷宫环境

**功能**：通用的迷宫环境框架，可以自定义迷宫布局

---

## 工具模块

### 11. `activations.py` - 激活函数

**功能**：提供各种神经网络激活函数

**可用函数**：
- `sigmoid_activation(x)`: Sigmoid函数，输出范围(0, 1)
- `tanh_activation(x)`: 双曲正切，输出范围(-1, 1)
- `relu_activation(x)`: ReLU，f(x) = max(0, x)
- `identity_activation(x)`: 恒等函数，f(x) = x
- `sin_activation(x)`: 正弦函数
- `abs_activation(x)`: 绝对值函数
- `gauss_activation(x)`: 高斯函数
- 等等...

**使用方法**：
```python
from pytorch_neat.activations import tanh_activation, sigmoid_activation

# 在创建网络时指定
net = RecurrentNet.create(
    genome, config,
    activation=tanh_activation  # 指定激活函数
)
```

---

### 12. `aggregations.py` - 聚合函数

**功能**：定义如何聚合多个输入信号

**可用函数**：
- `sum_aggregation`: 求和（最常用）
- `product_aggregation`: 乘积
- `max_aggregation`: 最大值
- `min_aggregation`: 最小值
- `mean_aggregation`: 平均值

---

### 13. `dask_helpers.py` - 分布式计算辅助

**功能**：使用Dask进行分布式并行评估

**用途**：
- 在集群上并行评估多个基因组
- 加速大规模实验
- 支持多机多核计算

---

## 使用示例

### 示例1：简单的CartPole任务

```python
# 见 examples/simple/main.py
# 使用递归神经网络解决倒立摆问题
# - 环境：CartPole-v1
# - 网络：RecurrentNet
# - 评估：MultiEnvEvaluator
```

### 示例2：自适应网络解决T-Maze

```python
# 见 examples/adaptive/main.py
# 使用自适应权重网络解决需要记忆的导航任务
# - 环境：TMazeEnv
# - 网络：AdaptiveLinearNet
# - 特点：权重可以动态更新，实现短期记忆
```

---

## 🔧 典型工作流程

### 1. 定义问题

```python
# 选择或创建环境
def make_env():
    return gym.make("CartPole-v1")
```

### 2. 选择网络类型

```python
# 简单问题 → RecurrentNet
# 需要记忆 → AdaptiveLinearNet
# 复杂问题 → AdaptiveNet

def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)
```

### 3. 定义动作选择

```python
def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return process_outputs(outputs)  # 转换为动作
```

### 4. 配置评估器

```python
evaluator = MultiEnvEvaluator(
    make_net, activate_net,
    make_env=make_env,
    batch_size=4,
    max_env_steps=1000
)
```

### 5. 运行NEAT

```python
# 加载配置
config = neat.Config(...)

# 定义评估函数
def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = evaluator.eval_genome(genome, config)

# 创建种群并运行
pop = neat.Population(config)
pop.add_reporter(LogReporter("log.json", evaluator.eval_genome))
winner = pop.run(eval_genomes, n_generations)
```

---

## 📊 性能优化建议

### 1. 批量处理
- 使用 `batch_size > 1` 同时评估多个环境
- 利用PyTorch的批量操作加速计算

### 2. 多进程并行
- 使用 `n_processes > 1` 并行评估多个基因组
- 注意内存占用和通信开销

### 3. 数据类型
- 默认使用 `float64`，可以改为 `float32` 节省内存
- 对于某些任务，精度降低影响不大

### 4. 网络大小
- 控制 `n_internal_steps` 避免过度计算
- 使用 `prune_empty=True` 移除无用连接

---

## 🐛 常见问题

### Q1: 训练速度慢？
**A**: 尝试增加 `batch_size` 和 `n_processes`，减少 `max_env_steps`

### Q2: 内存不足？
**A**: 减少 `batch_size` 和 `pop_size`，使用 `float32`

### Q3: 网络不收敛？
**A**: 
- 检查奖励设计是否合理
- 调整NEAT配置参数（变异率、种群大小等）
- 增加 `n_generations`
- 尝试不同的激活函数

### Q4: gymnasium API错误？
**A**: 本库已更新支持gymnasium（OpenAI Gym的后继者）
- `env.reset()` 返回 `(obs, info)`
- `env.step()` 返回 `(obs, reward, terminated, truncated, info)`

---

## 📚 进一步学习

### 推荐资源

1. **NEAT原理**：
   - [NEAT论文](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
   - [NEAT-Python文档](https://neat-python.readthedocs.io/)

2. **HyperNEAT**：
   - [HyperNEAT论文](http://eplex.cs.ucf.edu/papers/stanley_alife09.pdf)
   - 理解空间模式和CPPN

3. **自适应网络**：
   - [神经可塑性与元学习](https://arxiv.org/abs/1903.05134)
   - 理解权重如何动态变化

### 相关项目

- [neat-python](https://github.com/CodeReclaimers/neat-python): NEAT算法的Python实现
- [PyTorch](https://pytorch.org/): 深度学习框架
- [Gymnasium](https://gymnasium.farama.org/): 强化学习环境标准

---

## 📝 总结

**PyTorch-NEAT** 提供了：
- ✅ 多种神经网络类型（递归、自适应等）
- ✅ 高效的评估框架（批量、并行）
- ✅ 完整的训练工具（日志、报告）
- ✅ 测试环境（T-Maze系列）

**适合的任务**：
- 控制问题（CartPole、机器人等）
- 需要记忆的任务（T-Maze等）
- 快速适应任务
- 拓扑结构搜索

**开始使用**：
1. 查看 `examples/simple/main.py` 学习基础用法
2. 查看 `examples/adaptive/main.py` 学习高级功能
3. 根据自己的任务修改代码
4. 调整NEAT配置文件优化性能

祝你使用愉快！🚀

