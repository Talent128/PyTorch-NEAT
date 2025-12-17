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
多环境评估器：批量评估神经网络在多个环境实例中的表现

这个模块提供了一个评估器类，可以同时在多个环境实例中测试网络，
提高评估的效率和稳定性（通过平均多个运行的结果）。
"""

import numpy as np


class MultiEnvEvaluator:
    """
    多环境评估器类
    
    同时在多个环境实例中评估基因组，返回平均适应度。
    这种方法可以：
    1. 提高评估效率（批量处理）
    2. 提高评估稳定性（平均多次运行）
    3. 测试泛化能力（在不同环境配置下测试）
    """
    
    def __init__(self, make_net, activate_net, batch_size=5, max_env_steps=None, make_env=None, envs=None):
        """
        初始化多环境评估器
        
        Args:
            make_net: 函数，根据基因组创建神经网络
                     签名: make_net(genome, config, batch_size) -> network
            activate_net: 函数，激活网络获取动作
                         签名: activate_net(network, states) -> actions
            batch_size: 批量大小，即同时评估的环境数量
            max_env_steps: 每个episode的最大步数，None表示无限制
            make_env: 函数，创建环境实例（与envs二选一）
                     签名: make_env() -> gym.Env
            envs: 环境实例列表（与make_env二选一）
        """
        # 创建或使用提供的环境列表
        if envs is None:
            self.envs = [make_env() for _ in range(batch_size)]
        else:
            self.envs = envs
        
        # 保存网络创建和激活函数
        self.make_net = make_net
        self.activate_net = activate_net
        
        # 保存批量大小和最大步数
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps

    def eval_genome(self, genome, config, debug=False):
        """
        评估单个基因组的适应度
        
        在所有环境实例中运行基因组，返回平均奖励作为适应度。
        
        Args:
            genome: NEAT基因组
            config: NEAT配置对象
            debug: 是否启用调试输出（传递给activate_net）
            
        Returns:
            float: 平均适应度（所有环境的平均累积奖励）
            
        流程：
            1. 创建神经网络
            2. 重置所有环境
            3. 运行episode直到所有环境完成或达到最大步数
            4. 返回平均适应度
        """
        # 根据基因组创建神经网络
        net = self.make_net(genome, config, self.batch_size)

        # 初始化每个环境的累积适应度
        fitnesses = np.zeros(self.batch_size)
        
        # 重置所有环境，获取初始状态
        # gymnasium API: reset() 返回 (observation, info)
        states = [env.reset()[0] for env in self.envs]
        
        # 跟踪每个环境是否已完成
        dones = [False] * self.batch_size

        # 运行episode
        step_num = 0
        while True:
            step_num += 1
            
            # 检查是否达到最大步数限制
            if self.max_env_steps is not None and step_num == self.max_env_steps:
                break
            
            # 获取所有环境的动作
            if debug:
                # 调试模式：传递额外的调试信息
                actions = self.activate_net(
                    net, states, debug=True, step_num=step_num)
            else:
                # 正常模式
                actions = self.activate_net(net, states)
            #print(f"Step {step_num}, Actions: {actions}")
            # 确保动作数量与环境数量匹配
            assert len(actions) == len(self.envs)
            
            # 在每个环境中执行动作
            for i, (env, action, done) in enumerate(zip(self.envs, actions, dones)):
                if not done:  # 只更新未完成的环境
                    # 执行动作
                    # gymnasium API: step() 返回 (observation, reward, terminated, truncated, info)
                    state, reward, terminated, truncated, _ = env.step(action)
                    
                    # 合并terminated和truncated标志
                    done = terminated or truncated
                    
                    # 累加奖励
                    fitnesses[i] += reward
                    
                    # 更新状态（仅当未完成时）
                    if not done:
                        states[i] = state
                    
                    # 更新完成标志
                    dones[i] = done
            
            # 如果所有环境都完成，提前结束
            if all(dones):
                break

        # 返回所有环境的平均适应度
        return sum(fitnesses) / len(fitnesses)
