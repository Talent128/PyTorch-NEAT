"""
批量基因组评估器 - GPU优化版本

核心思想：
- GPU模式：单进程中同时评估多个基因组，避免spawn开销
- 通过扩展环境并行数 (n_genomes * trials_per_genome) 实现批量评估
- 环境只创建一次，复用于所有基因组
"""
import torch
import numpy as np
from typing import List, Tuple, Callable, Optional
from vmas import make_env


class BatchGenomeEvaluator:
    """
    批量基因组评估器
    
    对于GPU：在单个进程中批量评估多个基因组，避免多进程spawn开销
    核心优化：将 n_genomes * trials_per_genome 个环境实例合并为一个大批量环境
    """
    
    def __init__(
        self,
        scenario_name: str,
        task_config: dict,
        n_genomes_batch: int,
        trials_per_genome: int,
        n_steps: int,
        device: str,
        continuous_actions: bool = True,
    ):
        """
        初始化批量评估器
        
        Args:
            scenario_name: VMAS场景名称
            task_config: 任务配置字典
            n_genomes_batch: 每批评估的基因组数量
            trials_per_genome: 每个基因组的试验次数
            n_steps: 每次评估的最大步数
            device: 计算设备 ("cuda" 或 "cpu")
            continuous_actions: 是否使用连续动作
        """
        self.scenario_name = scenario_name
        self.task_config = task_config
        self.n_genomes_batch = n_genomes_batch
        self.trials_per_genome = trials_per_genome
        self.n_steps = n_steps
        self.device = device
        self.continuous_actions = continuous_actions
        
        # 总并行环境数 = 基因组数 * 每个基因组的试验数
        self.total_envs = n_genomes_batch * trials_per_genome
        
        # 创建批量环境（只创建一次！）
        self.env = self._make_batch_env()
        
        # 获取观测和动作维度
        obs = self.env.reset()
        self.obs_dim = obs[0].shape[-1]
        self.n_agents = len(obs)
        
        # 获取动作维度
        if hasattr(self.env.agents[0], 'action_space') and hasattr(self.env.agents[0].action_space, 'shape'):
            self.action_dim = self.env.agents[0].action_space.shape[0]
        else:
            self.action_dim = 2
    
    def _make_batch_env(self):
        """创建批量VMAS环境"""
        from dataclasses import asdict, is_dataclass
        
        if is_dataclass(self.task_config):
            env_kwargs = asdict(self.task_config)
        else:
            env_kwargs = dict(self.task_config)
        
        # 移除max_steps（由评估器控制）
        env_kwargs.pop('max_steps', None)
        
        env = make_env(
            scenario=self.scenario_name,
            num_envs=self.total_envs,  # 关键：创建大批量环境
            device=self.device,
            continuous_actions=self.continuous_actions,
            wrapper=None,
            **env_kwargs
        )
        return env
    
    def eval_genomes_batch(
        self,
        genomes: List[Tuple[int, object]],
        config,
        make_net: Callable,
        activate_net: Callable,
    ) -> List[Tuple[int, float]]:
        """
        批量评估多个基因组
        
        核心优化：
        1. 为每个基因组创建网络（但batch_size = trials_per_genome）
        2. 在同一个大环境中并行运行所有基因组
        3. 根据环境索引分配奖励到对应基因组
        
        Args:
            genomes: [(genome_id, genome), ...] 基因组列表
            config: NEAT配置
            make_net: 网络创建函数
            activate_net: 网络激活函数
            
        Returns:
            [(genome_id, fitness), ...] 适应度结果
        """
        n_genomes = len(genomes)
        
        # 如果基因组数量与预设不同，需要调整
        if n_genomes != self.n_genomes_batch:
            # 动态调整环境大小（较少见的情况）
            self.n_genomes_batch = n_genomes
            self.total_envs = n_genomes * self.trials_per_genome
            self.env = self._make_batch_env()
        
        # 为每个基因组创建网络
        # 注意：网络的 batch_size = trials_per_genome（对应该基因组的试验数）
        nets = []
        for genome_id, genome in genomes:
            net = make_net(genome, config, self.trials_per_genome)
            nets.append(net)
        
        # 重置环境
        obs = self.env.reset()  # obs[agent_idx]: (total_envs, obs_dim)
        
        # 初始化累积奖励: (n_genomes, trials_per_genome)
        cumulative_rewards = torch.zeros(
            n_genomes, self.trials_per_genome, 
            device=self.device
        )
        
        # 主循环
        for step in range(self.n_steps):
            # 为所有基因组计算动作
            # actions[agent_idx]: (total_envs, action_dim)
            actions = [None] * self.n_agents
            
            for agent_idx in range(self.n_agents):
                # 当前智能体在所有环境中的观测: (total_envs, obs_dim)
                agent_obs = obs[agent_idx]
                
                # 为每个基因组分别计算动作
                agent_actions_list = []
                for genome_idx, net in enumerate(nets):
                    # 获取该基因组对应的环境观测
                    # 环境索引: [genome_idx * trials_per_genome : (genome_idx + 1) * trials_per_genome]
                    start_idx = genome_idx * self.trials_per_genome
                    end_idx = start_idx + self.trials_per_genome
                    genome_obs = agent_obs[start_idx:end_idx]  # (trials_per_genome, obs_dim)
                    
                    # 使用该基因组的网络计算动作
                    genome_action = activate_net(
                        net, 
                        genome_obs, 
                        u_range=self.env.agents[agent_idx].u_range
                    )  # (trials_per_genome, action_dim)
                    
                    agent_actions_list.append(genome_action)
                
                # 合并所有基因组的动作: (total_envs, action_dim)
                actions[agent_idx] = torch.cat(agent_actions_list, dim=0)
            
            # 执行动作
            obs, rews, dones, info = self.env.step(actions)
            
            # 计算奖励: rewards (n_agents, total_envs)
            rewards = torch.stack(rews, dim=0)  # (n_agents, total_envs)
            
            # 计算全局奖励（所有智能体平均）
            global_reward = rewards.mean(dim=0)  # (total_envs,)
            
            # 将奖励分配到对应基因组
            # reshape: (n_genomes, trials_per_genome)
            reward_per_genome = global_reward.view(n_genomes, self.trials_per_genome)
            cumulative_rewards += reward_per_genome
        
        # 计算每个基因组的平均适应度
        # 对每个基因组的 trials_per_genome 次试验取平均
        mean_fitness = cumulative_rewards.mean(dim=1)  # (n_genomes,)
        
        # 返回结果
        results = []
        for i, (genome_id, genome) in enumerate(genomes):
            fitness = mean_fitness[i].item()
            results.append((genome_id, fitness))
        
        return results


class HybridEvaluator:
    """
    混合评估器：根据设备自动选择最优评估策略
    
    - CPU: 使用多进程并行（fork模式，开销小）
    - GPU: 使用单进程批量评估（避免spawn开销）
    """
    
    def __init__(
        self,
        experiment,  # Experiment实例
    ):
        self.experiment = experiment
        self.device = experiment.config.device
        self.is_cuda = 'cuda' in self.device.lower()
        
        if self.is_cuda:
            # GPU模式：创建批量评估器
            from dataclasses import asdict
            self.batch_evaluator = BatchGenomeEvaluator(
                scenario_name=experiment.scenario_name,
                task_config=experiment.task_config,
                n_genomes_batch=experiment.config.n_parallel,
                trials_per_genome=experiment.config.trials,
                n_steps=getattr(experiment.task_config, 'max_steps', 200),
                device=experiment.config.device,
                continuous_actions=experiment.config.continuous_actions,
            )
    
    def eval_genomes(self, genomes, config):
        """
        评估所有基因组
        
        Args:
            genomes: [(genome_id, genome), ...] 
            config: NEAT配置
        """
        if self.is_cuda:
            self._eval_genomes_gpu_batch(genomes, config)
        else:
            self._eval_genomes_cpu_parallel(genomes, config)
    
    def _eval_genomes_gpu_batch(self, genomes, config):
        """
        GPU批量评估：单进程，批量处理基因组
        """
        n_parallel = self.experiment.config.n_parallel
        
        # 分批处理
        for batch_start in range(0, len(genomes), n_parallel):
            batch_end = min(batch_start + n_parallel, len(genomes))
            batch_genomes = genomes[batch_start:batch_end]
            
            # 批量评估
            results = self.batch_evaluator.eval_genomes_batch(
                batch_genomes,
                config,
                make_net=self.experiment._make_net,
                activate_net=self.experiment._activate_net,
            )
            
            # 更新适应度
            genomes_dict = dict(genomes)
            for genome_id, fitness in results:
                genomes_dict[genome_id].fitness = fitness
    
    def _eval_genomes_cpu_parallel(self, genomes, config):
        """
        CPU多进程并行评估（使用fork模式，开销小）
        """
        import multiprocessing
        from .evaluator import GenomeEvaluator
        
        # 准备评估器参数
        evaluator_kwargs = {
            'make_net': self.experiment._make_net,
            'activate_net': self.experiment._activate_net,
            'make_env': self.experiment._make_env,
            'n_steps': getattr(self.experiment.task_config, 'max_steps', 200),
            'batch_size': self.experiment.config.trials,
            'render': False,
            'save_render': False,
            'video_dir': self.experiment.video_dir,
            'scenario_name': self.experiment.scenario_name
        }
        
        def eval_worker(args):
            genome_id, genome, config, kwargs = args
            evaluator = GenomeEvaluator(**kwargs)
            fitness = evaluator.eval_genome(genome, config)
            return genome_id, fitness
        
        args_list = [
            (genome_id, genome, config, evaluator_kwargs)
            for genome_id, genome in genomes
        ]
        
        n_parallel = self.experiment.config.n_parallel
        
        if n_parallel > 1:
            # CPU使用fork模式，开销更小
            # 注意：fork在Linux下可用，Windows需要spawn
            import platform
            if platform.system() == 'Linux':
                ctx = multiprocessing.get_context('fork')
            else:
                ctx = multiprocessing.get_context('spawn')
            
            with ctx.Pool(n_parallel) as pool:
                results = pool.map(eval_worker, args_list)
        else:
            results = [eval_worker(args) for args in args_list]
        
        # 更新适应度
        genomes_dict = dict(genomes)
        for genome_id, fitness in results:
            genomes_dict[genome_id].fitness = fitness

