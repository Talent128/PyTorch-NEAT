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
NEAT训练日志记录器

这个模块提供了一个报告器类，用于记录NEAT训练过程中的详细统计信息到JSON文件。
每一代的信息以单独的JSON行记录，便于后续分析、绘图和监控训练进度。

JSON文件格式：
每行包含一代的完整统计信息，包括适应度、网络结构、种群信息、时间消耗等。
"""

import json  # JSON序列化，用于保存日志
import time  # 时间测量，用于统计每代耗时
from pprint import pprint  # 美化打印，用于终端输出

import numpy as np  # 数值计算，用于统计均值和标准差
from neat.reporting import BaseReporter  # NEAT基础报告器类


class LogReporter(BaseReporter):
    """
    JSON日志报告器类
    
    继承自NEAT的BaseReporter，在训练过程中的关键时刻被调用，
    收集并记录各种统计信息到JSON文件。
    
    这个报告器会：
    1. 记录每代的适应度统计
    2. 评估最佳个体的验证性能
    3. 记录种群和物种信息
    4. 跟踪训练时间
    5. 记录最佳网络的结构信息
    """
    def __init__(self, fnm, eval_best, eval_with_debug=False):
        """
        初始化日志报告器
        
        Args:
            fnm (str): 日志文件名（会以追加模式打开）
            eval_best (callable): 评估函数，用于测试最佳个体
                                 签名: eval_best(genome, config, debug=False) -> fitness
            eval_with_debug (bool): 是否在评估最佳个体时启用调试输出
        """
        self.log = open(fnm, "a")  # 以追加模式打开日志文件
        self.generation = None  # 当前代数（未使用）
        self.generation_start_time = None  # 当前代开始时间
        self.generation_times = []  # 最近几代的耗时列表（用于计算平均值）
        self.num_extinctions = 0  # 累计物种灭绝次数
        self.eval_best = eval_best  # 保存评估函数
        self.eval_with_debug = eval_with_debug  # 保存调试标志
        self.log_dict = {}  # 当前代的日志字典

    def start_generation(self, generation):
        """
        在每代开始时调用
        
        记录代数并开始计时。
        
        Args:
            generation (int): 当前代数
        """
        self.log_dict["generation"] = generation  # 记录代数
        self.generation_start_time = time.time()  # 记录开始时间

    def end_generation(self, config, population, species_set):
        """
        在每代结束时调用
        
        记录种群统计信息、时间统计，并将所有信息写入日志文件。
        
        Args:
            config: NEAT配置对象
            population (dict): 当前种群 {genome_id: genome}
            species_set: 物种集合对象
        """
        # 记录种群大小
        ng = len(population)  # 计算种群中的个体数量
        self.log_dict["pop_size"] = ng  # 保存到日志字典

        # 记录物种数量
        ns = len(species_set.species)  # 计算当前物种数量
        self.log_dict["n_species"] = ns  # 保存到日志字典

        # 计算并记录本代耗时
        elapsed = time.time() - self.generation_start_time  # 计算经过的时间
        self.log_dict["time_elapsed"] = elapsed  # 保存本代耗时

        # 维护最近10代的耗时，用于计算平均值
        self.generation_times.append(elapsed)  # 添加当前耗时
        self.generation_times = self.generation_times[-10:]  # 只保留最近10个
        average = np.mean(self.generation_times)  # 计算平均耗时
        self.log_dict["time_elapsed_avg"] = average  # 保存平均耗时

        # 记录累计灭绝次数
        self.log_dict["n_extinctions"] = self.num_extinctions

        # 美化打印到终端（方便实时监控）
        pprint(self.log_dict)
        
        # 将日志字典序列化为JSON并写入文件（每行一个JSON对象）
        self.log.write(json.dumps(self.log_dict) + "\n")

    def post_evaluate(self, config, population, species, best_genome):
        """
        在评估完所有个体后调用
        
        计算适应度统计信息，评估最佳个体的验证性能，记录最佳网络结构。
        
        Args:
            config: NEAT配置对象
            population (dict): 当前种群 {genome_id: genome}
            species: 物种集合（未使用）
            best_genome: 本代最佳基因组
        """
        # pylint: disable=no-self-use
        
        # 收集所有个体的适应度
        fitnesses = [c.fitness for c in population.values()]
        
        # 计算适应度的统计量
        fit_mean = np.mean(fitnesses)  # 平均适应度
        fit_std = np.std(fitnesses)  # 适应度标准差

        # 记录适应度统计
        self.log_dict["fitness_avg"] = fit_mean
        self.log_dict["fitness_std"] = fit_std

        # 记录最佳适应度（训练集）
        self.log_dict["fitness_best"] = best_genome.fitness

        # 打印分隔线，突出显示最佳基因组信息
        print("=" * 50 + " Best Genome: " + "=" * 50)
        
        # 如果启用调试，打印基因组详细信息
        if self.eval_with_debug:
            print(best_genome)

        # 重新评估最佳个体（验证性能）
        # 这可以测试过拟合程度（训练适应度 vs 验证适应度）
        best_fitness_val = self.eval_best(
            best_genome, config, debug=self.eval_with_debug
        )
        self.log_dict["fitness_best_val"] = best_fitness_val

        # 获取并记录最佳网络的结构信息
        n_neurons_best, n_conns_best = best_genome.size()  # 获取神经元和连接数
        self.log_dict["n_neurons_best"] = n_neurons_best  # 神经元数量
        self.log_dict["n_conns_best"] = n_conns_best  # 连接数量

    def complete_extinction(self):
        """
        当一个物种完全灭绝时调用
        
        递增灭绝计数器。物种灭绝意味着该物种的所有成员都被淘汰，
        这可能发生在物种长期停滞或表现不佳时。
        """
        self.num_extinctions += 1  # 递增灭绝计数

    def found_solution(self, config, generation, best):
        """
        当找到满足适应度阈值的解时调用
        
        这是BaseReporter接口的一部分，但本实现中未使用。
        
        Args:
            config: NEAT配置对象
            generation (int): 找到解的代数
            best: 最佳基因组
        """
        pass  # 不需要额外操作

    def species_stagnant(self, sid, species):
        """
        当一个物种停滞不前时调用
        
        物种停滞意味着该物种在一定代数内没有适应度提升。
        这是BaseReporter接口的一部分，但本实现中未使用。
        
        Args:
            sid: 物种ID
            species: 物种对象
        """
        pass  # 不需要额外操作
