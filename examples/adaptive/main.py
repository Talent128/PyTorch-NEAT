# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

"""
Adaptive示例：使用自适应线性网络解决T-Maze记忆任务

这个示例展示了如何使用自适应权重的神经网络（通过HyperNEAT/CPPN生成）来解决
需要记忆能力的T-Maze导航任务。网络的权重可以在运行时根据输入和输出动态调整，
这使得网络具有短期记忆和学习能力。

关键特性：
- 自适应权重：权重在每个时间步根据神经元激活值更新
- 空间编码：使用坐标来表示神经元位置，利用空间结构
- CPPN：使用组合模式生成网络（CPPN）来生成权重更新规则
"""

import multiprocessing  # 多进程并行评估
import os  # 文件路径操作

import click  # 命令行参数解析
import neat  # NEAT算法核心库

# import torch  # PyTorch库（未使用，保留供参考）
import numpy as np  # 数值计算库

from pytorch_neat import t_maze  # T-Maze环境
from pytorch_neat.activations import tanh_activation  # tanh激活函数
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet  # 自适应线性网络
from pytorch_neat.multi_env_eval import MultiEnvEvaluator  # 多环境评估器
from pytorch_neat.neat_reporter import LogReporter  # 日志记录器

# 批量大小：同时评估4个环境以提高效率和稳定性
batch_size = 4

# 调试标志：启用时会打印网络状态和权重变化
DEBUG = True


def make_net(genome, config, _batch_size):
    """
    创建自适应线性神经网络
    
    使用空间坐标来定义神经元位置，CPPN会利用这些坐标生成权重更新规则。
    这种方法受到HyperNEAT的启发，利用几何拓扑来组织网络结构。
    
    Args:
        genome: NEAT基因组，编码CPPN网络
        config: NEAT配置对象
        _batch_size: 批量大小（未使用，使用全局batch_size）
        
    Returns:
        AdaptiveLinearNet: 自适应线性网络实例
        
    网络结构：
        - 输入层：4个神经元，位于不同坐标
          [-1, 0]：左侧传感器（墙壁检测）
          [0, 0]：前方传感器（墙壁检测）
          [1, 0]：右侧传感器（墙壁检测）
          [0, -1]：颜色传感器（奖励提示）
          
        - 输出层：3个神经元（动作）
          [-1, 0]：左转
          [0, 0]：前进
          [1, 0]：右转
    """
    # 输入神经元的2D空间坐标
    input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    
    # 输出神经元的2D空间坐标
    output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    
    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,  # 输入神经元坐标
        output_coords=output_coords,  # 输出神经元坐标
        weight_threshold=0.4,  # 权重阈值：小于此值的权重被设为0
        batch_size=batch_size,  # 批量大小
        activation=tanh_activation,  # 隐藏层激活函数
        output_activation=tanh_activation,  # 输出层激活函数
        device="cpu",  # 使用CPU计算
    )


def activate_net(net, states, debug=False, step_num=0):
    """
    激活网络并选择动作
    
    这个函数不仅前向传播计算输出，还会更新网络的自适应权重。
    在debug模式下会打印网络状态，帮助理解权重如何随时间演化。
    
    Args:
        net: 自适应线性网络实例
        states: 环境状态数组，形状为 (batch_size, 4)
        debug: 是否打印调试信息
        step_num: 当前步数（用于调试输出）
        
    Returns:
        numpy.ndarray: 选择的动作索引，形状为 (batch_size,)
                      0=左转, 1=前进, 2=右转
    """
    # 第一步时打印初始网络信息
    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)  # 打印权重更新规则（CPPN节点）
        print("W init: ", net.input_to_output[0])  # 打印初始权重矩阵
    
    # 前向传播：计算输出并更新自适应权重
    outputs = net.activate(states).numpy()
    
    # 每100步打印一次网络状态（用于观察学习过程）
    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])  # 第一个环境的输出
        print("Delta W: ", net.delta_w[0])  # 第一个环境的权重变化
        print("W: ", net.input_to_output[0])  # 第一个环境的当前权重
    
    # 选择输出值最大的动作（argmax）
    return np.argmax(outputs, axis=1)


@click.command()  # 将函数转换为命令行命令
@click.option("--n_generations", type=int, default=1000)  # 训练代数（默认1000）
@click.option("--n_processes", type=int, default=16)  # 并行进程数（默认1，单进程）
def run(n_generations, n_processes):
    """
    运行自适应网络的NEAT训练
    
    这个函数设置并运行完整的训练流程，包括：
    1. 加载配置
    2. 创建T-Maze环境（需要记忆能力的导航任务）
    3. 配置评估器
    4. 运行进化算法
    5. 评估最终获胜者
    
    Args:
        n_generations (int): 训练的代数，默认10000代
        n_processes (int): 并行评估的进程数，>1时启用多进程
    
    Returns:
        int: 实际训练的代数
    """
    # 加载NEAT配置文件
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,  # 标准基因组
        neat.DefaultReproduction,  # 标准繁殖
        neat.DefaultSpeciesSet,  # 标准物种集
        neat.DefaultStagnation,  # 标准停滞检测
        config_path,  # 配置文件路径
    )

    # 创建4个T-Maze环境，奖励位置不同以测试泛化能力
    # init_reward_side: 0=左侧有高奖励, 1=右侧有高奖励
    # n_trials: 每个episode有100次试验
    envs = [t_maze.TMazeEnv(init_reward_side=i, n_trials=100) for i in [1, 0, 1, 0]]

    # 创建多环境评估器
    evaluator = MultiEnvEvaluator(
        make_net,  # 网络创建函数
        activate_net,  # 网络激活函数
        envs=envs,  # 环境列表
        batch_size=batch_size,  # 批量大小
        max_env_steps=1000  # 每个episode的最大步数
    )

    # 根据进程数选择评估策略
    if n_processes > 1:
        # 多进程模式：并行评估多个基因组以加速训练
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            """
            并行评估所有基因组
            
            使用进程池同时评估多个基因组，显著加速训练。
            适合CPU密集型任务。
            """
            # 使用starmap并行执行评估
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            # 将适应度分配给对应的基因组
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:
        # 单进程模式：串行评估，便于调试

        def eval_genomes(genomes, config):
            """
            串行评估所有基因组
            
            逐个评估基因组，适合调试。每100个基因组启用一次debug输出。
            """
            for i, (_, genome) in enumerate(genomes):
                try:
                    # 每100个基因组启用一次调试输出
                    genome.fitness = evaluator.eval_genome(
                        genome, config, debug=DEBUG and i % 100 == 0
                    )
                except Exception as e:
                    # 出错时打印基因组信息，便于调试
                    print(genome)
                    raise e

    # 创建初始种群
    pop = neat.Population(config)
    
    # 添加统计报告器：收集训练统计信息
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # 添加标准输出报告器：在终端显示进度
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    
    # 添加JSON日志报告器：保存详细训练日志到log.json
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    # 运行进化算法，返回最佳基因组（获胜者）
    winner = pop.run(eval_genomes, n_generations)

    # 打印获胜基因组的详细信息
    print(winner)
    
    # 评估获胜者的最终表现
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    
    # 返回实际训练的代数
    generations = reporter.generation + 1
    return generations


if __name__ == "__main__":
    # 程序入口点
    # pylint: disable=no-value-for-parameter
    # click会自动处理命令行参数
    run()
