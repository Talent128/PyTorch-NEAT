"""
NEAT进化实验模块

管理整个NEAT进化训练流程
"""
from dataclasses import dataclass
from typing import Optional
import os
import pickle
import time
import multiprocessing
import tempfile

import neat
import torch
import numpy as np
from vmas import make_env

from pytorch_neat.recurrent_net import RecurrentNet
from pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from pytorch_neat.adaptive_net import AdaptiveNet
from pytorch_neat.activations import str_to_activation

from .evaluator import GenomeEvaluator
from .neat_reporter import LogReporter
from .utils import (
    seed_everything,
    generate_circular_coords,
    generate_grid_coords,
    load_neat_config_with_substitution,
)


@dataclass
class ExperimentConfig:
    """实验配置类"""
    generations: int
    trials: int
    n_parallel: int

    device: str = "cpu"
    continuous_actions: bool = True
    overwrite: bool = False

    render: bool = False
    save_render: bool = False
    train_render: bool = False  # 训练时是否渲染（默认False，仅在result_of_experiment中渲染）
    show_gen: int = -1
    collect_results: bool = True

    checkpt_freq: int = 1
    results_dir: Optional[str] = None  # 结果目录，null表示自动生成


class Experiment:
    """NEAT进化实验类"""
    
    def __init__(
        self,
        task_name: str,
        algorithm_name: str,
        algorithm_config: dict,
        task_config: dict,
        experiment_config: ExperimentConfig,
        seed: int = 0,
    ):
        """初始化实验
        
        Args:
            task_name (str): 任务名称，如 "vmas/transport"
            algorithm_name (str): 算法名称，如 "recurrent"
            algorithm_config (dict): 算法配置字典
            task_config (dict): 任务配置字典
            experiment_config (ExperimentConfig): 实验配置
            seed (int): 随机种子
        """
        self.task_name = task_name
        self.algorithm_name = algorithm_name
        self.algorithm_config = algorithm_config
        self.task_config = task_config
        self.config = experiment_config
        self.seed = seed
        
        # 设置随机种子
        seed_everything(seed)
        
        # 解析任务名称
        self.env_name, self.scenario_name = task_name.split("/")
        assert self.env_name == "vmas", f"仅支持vmas环境，当前: {self.env_name}"

        # 获取n_agents参数（用于路径命名）
        self.n_agents = getattr(task_config, 'n_agents', 'default')

        # 设置结果目录
        if self.config.results_dir is None:
            # 格式: results/n_agents-task_name-algorithm_name-seed{seed}
            self.config.results_dir = f"results/{self.n_agents}-{self.scenario_name}-{algorithm_name}-seed{seed}"

        # 创建结果目录及子文件夹
        os.makedirs(self.config.results_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.config.results_dir, "checkpoints")
        self.log_dir = os.path.join(self.config.results_dir, "logs")
        self.video_dir = os.path.join(self.config.results_dir, "videos")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # 创建测试环境以获取观测和动作维度
        test_env = self._make_env(num_envs=1)
        obs = test_env.reset()
        self.obs_dim = obs[0].shape[-1]
        # 从action_space或者通过执行一步获取动作维度
        if hasattr(test_env.agents[0], 'action_space') and hasattr(test_env.agents[0].action_space, 'shape'):
            self.action_dim = test_env.agents[0].action_space.shape[0]
        else:
            # 对于连续动作，动作维度通常是2（2D运动）
            # 可以通过agents[0].action的属性获取
            # vmas中，u是力/速度控制，通常是2D
            self.action_dim = 2  # 默认2D运动
        
        print(f"\n{'='*60}")
        print(f"任务: {task_name}")
        print(f"算法: {algorithm_name}")
        print(f"观测维度: {self.obs_dim}")
        print(f"动作维度: {self.action_dim}")
        print(f"智能体数量: {self.n_agents}")
        print(f"并行环境: {self.config.trials}")
        print(f"并行基因组: {self.config.n_parallel}")
        print(f"设备: {self.config.device}")
        print(f"{'='*60}\n")

        # 加载NEAT配置
        self.neat_config = self._load_neat_config()

        # 初始化种群
        self.population = None
        self.generation = 0
        
        # 设置multiprocessing启动方法（CUDA需要spawn）
        if self.config.device == "cuda" and self.config.n_parallel > 1:
            try:
                multiprocessing.set_start_method('spawn', force=True)

            except RuntimeError:
                # 如果已经设置过，忽略
                pass

    def _load_neat_config(self):
        """加载NEAT配置，替换输入输出维度"""
        cfg_path = self.algorithm_config.neat_config_path

        # 将配置文件保存到results目录
        neat_cfg_filename = os.path.basename(cfg_path)
        permanent_cfg_path = os.path.join(self.config.results_dir, neat_cfg_filename)

        # 如果配置文件已存在，直接使用
        if os.path.exists(permanent_cfg_path):
            print(f"使用已有的NEAT配置: {permanent_cfg_path}")
            temp_cfg_path = permanent_cfg_path
        else:
            # 创建配置文件，替换占位符
            temp_cfg_path = load_neat_config_with_substitution(
                cfg_path, self.obs_dim, self.action_dim, permanent_cfg_path
            )
            print(f"NEAT配置已保存至: {permanent_cfg_path}")

        # 加载NEAT配置
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_cfg_path
        )

        return neat_config

    def _make_env(self, num_envs=None):
        """创建VMAS环境

        Args:
            num_envs (int, optional): 并行环境数量，默认使用config.trials

        Returns:
            vmas.Env: VMAS环境实例
        """
        if num_envs is None:
            num_envs = self.config.trials

        # 准备环境参数（task_config是dataclass）
        from dataclasses import asdict
        env_kwargs = asdict(self.task_config)
        # 移除max_steps（由experiment控制）
        max_steps = env_kwargs.pop('max_steps', 200)

        env = make_env(
            scenario=self.scenario_name,
            num_envs=num_envs,
            device=self.config.device,
            continuous_actions=self.config.continuous_actions,
            wrapper=None,
            **env_kwargs
        )

        return env

    def _make_net(self, genome, config, batch_size):
        """根据算法类型创建神经网络

        Args:
            genome: NEAT基因组
            config: NEAT配置
            batch_size (int): 批量大小（通常等于experiment.trials，即并行环境数）

        Returns:
            网络实例

        Note:
            - batch_size: 算法配置中的batch_size会被此处覆盖，使用experiment.trials
            - device: 使用experiment.device（会覆盖算法配置中的device）
        """
        device = self.config.device  # 使用实验配置中的device

        if self.algorithm_name == "recurrent":
            # RecurrentNet
            activation = str_to_activation.get(
                getattr(self.algorithm_config, 'activation', 'sigmoid')
            )
            return RecurrentNet.create(
                genome,
                config,
                batch_size=batch_size,
                activation=activation,
                use_current_activs=getattr(self.algorithm_config, 'use_current_activs', False),
                n_internal_steps=getattr(self.algorithm_config, 'n_internal_steps', 1),
                prune_empty=getattr(self.algorithm_config, 'prune_empty', True),
                dtype=getattr(torch, getattr(self.algorithm_config, 'dtype', 'float32')),
                device=device,
            )

        elif self.algorithm_name == "adaptive_linear":
            # AdaptiveLinearNet
            # 生成坐标
            input_coords = getattr(self.algorithm_config, 'input_coords', None)
            if input_coords is None:
                input_coords = generate_circular_coords(self.obs_dim)

            output_coords = getattr(self.algorithm_config, 'output_coords', None)
            if output_coords is None:
                output_coords = generate_circular_coords(self.action_dim)

            activation = str_to_activation.get(
                getattr(self.algorithm_config, 'activation', 'tanh')
            )
            cppn_activation = str_to_activation.get(
                getattr(self.algorithm_config, 'cppn_activation', 'identity')
            )

            return AdaptiveLinearNet.create(
                genome,
                config,
                input_coords=input_coords,
                output_coords=output_coords,
                weight_threshold=getattr(self.algorithm_config, 'weight_threshold', 0.3),
                weight_max=getattr(self.algorithm_config, 'weight_max', 3.0),
                activation=activation,
                cppn_activation=cppn_activation,
                batch_size=batch_size,
                device=device,
            )

        elif self.algorithm_name == "adaptive":
            # AdaptiveNet
            # 生成坐标
            input_coords = getattr(self.algorithm_config, 'input_coords', None)
            if input_coords is None:
                input_coords = generate_circular_coords(self.obs_dim)

            hidden_coords = getattr(self.algorithm_config, 'hidden_coords', None)
            if hidden_coords is None:
                # 默认隐藏层节点数
                n_hidden = max(4, (self.obs_dim + self.action_dim) // 2)
                hidden_coords = generate_grid_coords(n_hidden)

            output_coords = getattr(self.algorithm_config, 'output_coords', None)
            if output_coords is None:
                output_coords = generate_circular_coords(self.action_dim)

            activation = str_to_activation.get(
                getattr(self.algorithm_config, 'activation', 'tanh')
            )

            return AdaptiveNet.create(
                genome,
                config,
                input_coords=input_coords,
                hidden_coords=hidden_coords,
                output_coords=output_coords,
                weight_threshold=getattr(self.algorithm_config, 'weight_threshold', 0.25),
                activation=activation,
                batch_size=batch_size,
                device=device,
            )

        else:
            raise ValueError(f"未知算法: {self.algorithm_name}")

    def _activate_net(self, net, obs, u_range):
        """激活网络获取动作

        Args:
            net: 神经网络
            obs (torch.Tensor): 观测，shape=(batch_size, obs_dim)
            u_range (torch.Tensor): 动作范围 [min, max]

        Returns:
            torch.Tensor: 动作，shape=(batch_size, action_dim)
        """
        # 确保obs在正确的设备上
        if isinstance(obs, torch.Tensor):
            obs = obs.to(self.config.device)

        # 激活网络
        with torch.no_grad():
            output = net.activate(obs)

        # 将输出映射到动作空间
        # 假设使用tanh激活，输出范围是[-1, 1]
        # 需要映射到u_range
        # u_range可能是float（表示[-u_range, u_range]）或tensor/列表[min, max]
        if isinstance(u_range, (list, tuple)):
            u_min = u_range[0]
            u_max = u_range[1]
        elif isinstance(u_range, torch.Tensor):
            if u_range.numel() == 1:
                # 单个值
                u_min = -u_range.item()
                u_max = u_range.item()
            else:
                u_min = u_range[0]
                u_max = u_range[1]
        else:
            # u_range是一个标量float
            u_min = -u_range
            u_max = u_range

        # Clip到[-1, 1]（防止激活函数输出超出范围）
        output = torch.clamp(output, -1.0, 1.0)

        # 线性映射: [-1, 1] -> [u_min, u_max]
        action = (output + 1.0) / 2.0 * (u_max - u_min) + u_min

        return action

    def _eval_genome_worker(self, args):
        """评估基因组的工作函数

        Args:
            args: (genome_id, genome, config, evaluator_kwargs)

        Returns:
            (genome_id, fitness)
        """
        genome_id, genome, config, evaluator_kwargs = args

        # 创建评估器
        evaluator = GenomeEvaluator(**evaluator_kwargs)

        # 评估基因组，传递genome_id用于视频命名
        fitness = evaluator.eval_genome(genome, config, debug=False)

        return genome_id, fitness

    def _get_existing_fitness(self, genome, config, debug=False):
        """直接返回基因组已有的适应度（用于 LogReporter，避免重复评估）

        Args:
            genome: NEAT基因组
            config: NEAT配置
            debug: 是否启用调试输出（未使用）

        Returns:
            float: 基因组的已有适应度
        """
        return genome.fitness

    def _eval_genomes(self, genomes, config):
        """评估所有基因组（并行）

        Args:
            genomes: 基因组列表 [(genome_id, genome), ...]
            config: NEAT配置
        """
        # 准备评估器参数
        evaluator_kwargs = {
            'make_net': self._make_net,
            'activate_net': self._activate_net,
            'make_env': self._make_env,
            'n_steps': getattr(self.task_config, 'max_steps', 200),
            'batch_size': self.config.trials,
            'render': self.config.train_render,  # 训练时根据train_render参数决定是否渲染
            'save_render': False,   #训练时不保存视频(若每个基因都保存太多了，因而当前保存命名没有区分基因id)
            'video_dir': self.video_dir,
            'generation': self.generation,
            'scenario_name': self.scenario_name
        }

        # 准备多进程参数
        args_list = [
            (genome_id, genome, config, evaluator_kwargs)
            for genome_id, genome in genomes
        ]

        # 并行评估
        if self.config.n_parallel > 1:
            # 使用spawn方法创建进程池（CUDA兼容）
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(self.config.n_parallel) as pool:
                results = pool.map(self._eval_genome_worker, args_list)
        else:
            # 单进程（便于调试）
            results = [self._eval_genome_worker(args) for args in args_list]

        # 更新适应度
        for genome_id, fitness in results:
            genome = dict(genomes)[genome_id]
            genome.fitness = fitness

    def train(self):
        """训练NEAT种群"""
        print(f"\n开始训练 - 进化{self.config.generations}代\n")

        # 检查是否恢复
        if not self.config.overwrite:      
            print('RESTORING')
            checkpoint_file = self._find_latest_checkpoint(self.checkpoint_dir)
            if checkpoint_file is not None:
                print(f"从检查点恢复: {checkpoint_file}")
                self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
                self.generation = int(checkpoint_file.split('-')[-1])
                print(f"从第 {self.generation} 代继续训练\n")

        # 创建新种群
        if self.population is None:
            self.population = neat.Population(self.neat_config)
            self.generation = 0

        # 添加报告器
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)

        # 添加检查点
        checkpointer = neat.Checkpointer(
            self.config.checkpt_freq,
            filename_prefix=os.path.join(self.checkpoint_dir, 'neat-checkpoint-')
        )
        self.population.add_reporter(checkpointer)

        # 添加日志记录器（如果启用了collect_results）
        if self.config.collect_results:
            log_file = os.path.join(self.log_dir, "log.json")
            logger = LogReporter(log_file, self._get_existing_fitness, eval_with_debug=False)
            self.population.add_reporter(logger)
            print(f"已启用日志记录: {log_file}")

        # 运行进化
        start_time = time.time()
        winner = self.population.run(
            self._eval_genomes,
            self.config.generations - self.generation
        )
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("训练统计")
        print(f"{'='*60}")
        print(
                f"耗时: {elapsed_time/60:.2f} 分钟"
                f"共 {self.config.generations} 代"
                f"最佳适应度: {winner.fitness:.4f}"
            )
        print(f"{'='*60}")


        return winner

    def run(self):
        """运行实验：训练（如需要）+ 展示结果"""
        # 检查是否需要训练
        checkpoint_file = self._find_latest_checkpoint(self.checkpoint_dir)
        
        # 如果没有检查点，或者指定了代数但未完成训练
        if checkpoint_file is None:
            # 没有检查点，需要从头开始训练
            print("未找到检查点，将从头开始训练")
            self.train()
        else:
            # 有检查点，检查是否需要继续训练
            last_gen = int(checkpoint_file.split('-')[-1])
            if last_gen < self.config.generations - 1:
                print(f"检查点代数({last_gen}) < 目标代数({self.config.generations})，继续训练")
                self.train()
            else:
                print(f"训练已完成 (检查点代数: {last_gen})")
        
        # 展示实验结果
        self.result_of_experiment()

    def _find_latest_checkpoint(self, checkpoint_dir):
        """查找最新的检查点文件

        Args:
            checkpoint_dir (str): 检查点目录

        Returns:
            str or None: 最新检查点文件路径
        """
        if not os.path.exists(checkpoint_dir):
            return None

        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]
        if not checkpoints:
            return None

        # 按代数排序
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        return os.path.join(checkpoint_dir, checkpoints[-1])


    ####################################################################################################################
    # output functions
    ####################################################################################################################
    def result_of_experiment(self):
        """展示指定代数的最优个体结果
        
        根据 self.config.show_gen 参数：
        - show_gen >= 0: 展示指定代数的最优个体
        - show_gen < 0: 展示最后一代的最优个体（默认）
        """
        print(f"\n{'='*60}")
        print("展示实验结果")
        print(f"{'='*60}\n")
        
        # 查找检查点
        if not os.path.exists(self.checkpoint_dir):
            raise FileNotFoundError(f"检查点目录不存在: {self.checkpoint_dir}")
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('neat-checkpoint-')]
        if not checkpoints:
            raise FileNotFoundError(f"未找到检查点文件，请先训练模型")
        
        # 按代数排序
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        
        # 确定要展示的代数
        if self.config.show_gen >= 0:
            # 指定代数
            target_gen = self.config.show_gen
            checkpoint_name = f"neat-checkpoint-{target_gen}"
            if checkpoint_name not in checkpoints:
                raise ValueError(f"未找到第 {target_gen} 代的检查点")
            checkpoint_file = os.path.join(self.checkpoint_dir, checkpoint_name)
        else:
            # 最后一代
            checkpoint_file = os.path.join(self.checkpoint_dir, checkpoints[-1])
            target_gen = int(checkpoints[-1].split('-')[-1])
        
        print(f"加载第 {target_gen} 代的检查点: {checkpoint_file}")
        
        # 加载检查点
        population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # 获取最佳个体
        best_genome = None
        best_fitness = float('-inf')
        for genome_id, genome in population.population.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
        
        if best_genome is None:
            raise ValueError("未找到有效的最佳个体")
        
        print(f"最佳个体适应度: {best_fitness:.4f}\n")
        
        # 评估最佳个体（用于统计和渲染）
        print("评估最佳个体...")
        start_time = time.time()
        
        # 创建评估器（启用渲染如果设置了render参数）
        evaluator_kwargs = {
            'make_net': self._make_net,
            'activate_net': self._activate_net,
            'make_env': self._make_env,
            'n_steps': getattr(self.task_config, 'max_steps', 200),
            'batch_size': self.config.trials,
            'render': self.config.render,
            'save_render': self.config.save_render,
            'video_dir': self.video_dir,
            'generation': target_gen,
            'scenario_name': self.scenario_name
        }
        
        evaluator = GenomeEvaluator(**evaluator_kwargs)
        fitness = evaluator.eval_genome(best_genome, self.neat_config, debug=True)
        
        elapsed_time = time.time() - start_time
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print("评估统计")
        print(f"{'='*60}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"步数: {getattr(self.task_config, 'max_steps', 200)}")
        print(f"并行环境数: {self.config.trials}")
        print(f"设备: {self.config.device}")
        print(f"平均适应度: {fitness:.4f}")
        print(f"{'='*60}\n")
        