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
Simple示例：使用NEAT算法训练递归神经网络解决CartPole平衡问题

这是一个基础示例，展示如何使用PyTorch-NEAT库训练神经网络来控制倒立摆（CartPole）。
NEAT (NeuroEvolution of Augmenting Topologies) 是一种进化算法，可以同时优化网络的拓扑结构和权重。
"""

import os  # 用于文件路径操作

import click  # 命令行参数解析库
import gymnasium as gym  # OpenAI Gym环境库，提供强化学习环境
import neat  # NEAT算法核心库
import torch  # PyTorch库

from pytorch_neat.multi_env_eval import MultiEnvEvaluator  # 多环境评估器
from pytorch_neat.neat_reporter import LogReporter  # 日志记录器
from pytorch_neat.recurrent_net import RecurrentNet  # 递归神经网络

# 每个episode的最大步数，防止无限运行
max_env_steps = 200

# 全局设备变量，用于存储GPU/CPU设备
_device = 'cuda:3'


def make_env():
    """
    创建CartPole环境
    
    CartPole-v1是一个经典的强化学习问题：
    - 目标：通过左右移动小车来保持杆子竖直
    - 观测空间：4维（小车位置、小车速度、杆角度、杆角速度）
    - 动作空间：2维（向左或向右推小车）
    
    Returns:
        gym.Env: CartPole-v1环境实例
    """
    return gym.make("CartPole-v1")


def make_net(genome, config, bs):
    """
    根据NEAT基因组创建递归神经网络
    
    这个函数将NEAT进化出的基因组转换为可执行的PyTorch神经网络。
    基因组编码了网络的拓扑结构（节点和连接）以及权重。
    
    Args:
        genome: NEAT基因组，包含网络拓扑和权重信息
        config: NEAT配置对象
        bs: 批量大小（batch size）
        
    Returns:
        RecurrentNet: 递归神经网络实例
    """
    return RecurrentNet.create(genome, config, bs, device=_device)


def activate_net(net, states):
    """
    激活神经网络并将输出转换为动作
    
    这个函数接收网络和状态，返回要执行的动作。
    对于CartPole，输出需要是0（向左）或1（向右）。
    
    Args:
        net: 神经网络实例
        states: 环境状态数组，形状为 (batch_size, n_inputs)
        
    Returns:
        numpy.ndarray: 动作数组，形状为 (batch_size,)，值为0或1
        
    实现细节：
        - net.activate(states) 返回网络输出
        - outputs[:, 0] 取第一个输出神经元的值
        - > 0.5 将连续值转换为布尔值
        - .astype(int) 将布尔值转换为整数动作（0或1）
    """
    #print(f"Activating net with states: {states}")
    outputs = net.activate(states).detach().cpu().numpy()
    return (outputs[:, 0] > 0.5).astype(int)


@click.command()  # 将函数转换为命令行命令
@click.option("--n_generations", type=int, default=100)  # 命令行参数：训练代数
@click.option("--device", type=str, default=None, 
              help="计算设备: 'cuda', 'cuda:0', 'cpu'等。默认自动选择")
def run(n_generations, device):
    """
    运行NEAT训练过程
    
    这是主训练循环，执行以下步骤：
    1. 加载NEAT配置
    2. 创建评估器
    3. 创建种群
    4. 添加报告器（用于监控训练进度）
    5. 运行进化过程
    
    Args:
        n_generations (int): 要训练的代数，默认100代
        device (str): 计算设备，如 'cuda', 'cuda:0', 'cpu'
    """
    # 设置全局设备变量
    global _device
    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(device)
    print(f"使用设备: {_device}")
    
    # 加载配置文件，配置文件定义了NEAT的所有超参数
    # 包括种群大小、变异率、交叉率、激活函数等
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,  # 基因组类型：标准基因组
        neat.DefaultReproduction,  # 繁殖策略：标准繁殖
        neat.DefaultSpeciesSet,  # 物种划分：标准物种集
        neat.DefaultStagnation,  # 停滞检测：标准停滞检测
        config_path,  # 配置文件路径
    )

    # 创建多环境评估器
    # 评估器负责在环境中测试每个基因组，并返回适应度分数
    evaluator = MultiEnvEvaluator(
        make_net,  # 网络创建函数
        activate_net,  # 网络激活函数
        make_env=make_env,  # 环境创建函数
        max_env_steps=max_env_steps  # 每个episode的最大步数
    )

    def eval_genomes(genomes, config):
        """
        评估所有基因组的适应度
        
        这个函数会被NEAT在每一代调用，用于评估种群中所有个体的表现。
        
        Args:
            genomes: [(genome_id, genome), ...] 基因组列表
            config: NEAT配置对象
        """
        for _, genome in genomes:
            # 评估每个基因组并设置其适应度
            # 适应度越高，基因组在下一代中被选中的概率越大
            genome.fitness = evaluator.eval_genome(genome, config)

    # 创建初始种群
    pop = neat.Population(config)
    
    # 添加统计报告器：收集并显示统计信息
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    # 添加标准输出报告器：在终端打印训练进度
    reporter = neat.StdOutReporter(True)  # True表示显示物种信息
    pop.add_reporter(reporter)
    
    # 添加日志报告器：将详细信息保存到neat.log文件
    logger = LogReporter("neat.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    # 运行进化过程指定的代数
    # 每一代：评估 -> 选择 -> 交叉 -> 变异 -> 新种群
    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    # 程序入口点
    # pylint: disable=no-value-for-parameter 禁用该行的pylint检查
    # 因为click会自动处理参数，不需要手动传入
    run()
