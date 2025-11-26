import numpy as np
import torch
from vmas.simulator.utils import save_video  # 视频保存工具
import time
import os

class GenomeEvaluator:
    def __init__(self, make_net, activate_net, batch_size=5, n_steps=None, make_env=None, 
                 render=False, save_render=False, video_dir=None, generation=-1, scenario_name=""):
        # 创建或使用提供的环境列表
        self.env = make_env() 
        
        # 保存网络创建和激活函数
        self.make_net = make_net
        self.activate_net = activate_net
        
        # 保存批量大小和最大步数
        self.batch_size = batch_size
        self.n_steps = n_steps

        # 渲染相关
        self.render = render
        self.save_render = save_render
        self.video_dir = video_dir
        self.generation = generation
        self.scenario_name = scenario_name

        assert not (save_render and not render), "要保存视频，必须启用渲染（render=True）"

    def eval_genome(self, genome, config, debug=False):
        """
        评估单个基因组的适应度
        """
        # 根据基因组创建神经网络
        net = self.make_net(genome, config, self.batch_size)

        # ========== 准备渲染和统计 ==========
        frame_list = []  # 存储渲染帧，用于创建GIF或视频
        init_time = time.time()  # 记录开始时间
        step = 0  # 步数计数器

        total_reward = 0  # 累计总奖励
        
        # 重置环境，获取初始观测,obs是一个列表，每个元素对应一个智能体的观测,每个观测的形状: (n_envs, obs_dim)
        obs = self.env.reset()
        
        # ========== 主循环 ==========
        for _ in range(self.n_steps):
            step += 1
            
            # ===== 为每个智能体计算动作 =====
            # 创建动作列表，初始化为None
            actions = [None] * len(obs)
            
            # 遍历所有智能体的观测
            for i in range(len(obs)):
                # 使用策略计算动作
                # obs[i]: 第i个智能体在所有环境中的观测 (n_envs, obs_dim)
                # u_range: 动作的允许范围（例如 [-1, 1]）,u_range=env.agents[i].u_range
                # 返回: (n_envs, action_dim) 的动作张量
                #print(f"vmas-Obs for agent {i} at step {step}: {obs[i]}")
                actions[i] = self.activate_net(
                    net,
                    obs[i],
                    u_range=self.env.agents[i].u_range
                )
            #print(f"Step {step}, neat—Actions: {actions}")

            # ===== 执行动作，获取下一步状态 =====
            # step()返回四个值：
            # - obs: 新观测列表
            # - rews: 奖励列表，每个元素形状 (n_envs,)
            # - dones: 终止标志，形状 (n_envs,)
            # - info: 额外信息字典列表
            obs, rews, dones, info = self.env.step(actions)
            
            # ===== 计算和累积奖励 =====
            # 将奖励列表堆叠成张量
            # rewards 形状: (n_envs, n_agents)
            rewards = torch.stack(rews, dim=1)
            #print(f"Step {step}, Rewards: {rewards}")
            
            # 计算每个环境的全局奖励（所有智能体的平均）
            # global_reward 形状: (n_envs,)
            global_reward = rewards.mean(dim=1)
            #print(f"Step {step}, global_reward: {global_reward}")
            # 计算所有环境的平均奖励（标量）
            mean_global_reward = global_reward.mean(dim=0).item()
            #print(f"Step {step}, mean_global_reward: {mean_global_reward}")
            # 累积奖励
            total_reward += mean_global_reward
            #print(f"Step {step}, total_reward: {total_reward}")
            # =====  渲染（如果启用） =====
            if self.render:
                # 渲染当前帧
                # mode="rgb_array": 返回numpy数组而不是显示窗口
                # agent_index_focus=None: 相机不跟随特定智能体
                # visualize_when_rgb=True: 在RGB模式下显示可视化信息
                frame_list.append(
                    self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                )

        # 计算总耗时
        total_time = time.time() - init_time
        
        # 如果需要保存视频
        if self.render and self.save_render:
            # 视频文件名包含代数和基因组ID
            video_basename = f"{self.scenario_name}_gen{self.generation}_{total_reward:.2f}" #视频命名规则:场景名_代数_总奖励
            
            # save_video会在当前目录保存，需要切换到video_dir
            old_cwd = os.getcwd()
            try:
                os.chdir(self.video_dir)        #save_video会在当前目录保存，需要切换到video_dir目录保存视频
                # fps = 1 / dt，dt是仿真时间步长
                fps = int(1 / self.env.scenario.world.dt)
                save_video(video_basename, frame_list, fps)
                video_path = os.path.join(self.video_dir, f"{video_basename}.mp4")
                if debug:
                    print(f"视频已保存至: {video_path}")
            finally:
                os.chdir(old_cwd)
        
        """
        print(
            f"耗时: {total_time}秒，运行了 {n_steps} 步，"
            f"共 {n_envs} 个并行环境，设备: {device}\n"
            f"平均总奖励: {total_reward}"
        )
        """

        # 返回所有环境的平均适应度
        return total_reward
