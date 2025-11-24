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
自适应线性网络模块 (AdaptiveLinearNet)

这是一个无隐藏层的自适应神经网络，直接将输入连接到输出。
虽然结构简单，但通过自适应权重机制，它可以在运行时学习和适应。

网络结构：
    输入层 --[W(t), 自适应]--> 输出层

自适应机制：
    - 初始权重由CPPN根据神经元坐标生成
    - 每个时间步，权重根据输入输出的激活值更新：
      W(t+1) = W(t) + delta_w_node(坐标, input(t), output(t), W(t))
    - 这使网络能够实现短期记忆和快速适应

与其他网络的区别：
    - 无隐藏层，计算最快
    - 只有输入到输出的权重是自适应的
    - 适合需要快速适应但不需要复杂表示的任务

适用场景：
    - 需要短期记忆的简单任务（如T-Maze）
    - 需要快速在线学习的任务
    - 资源受限的环境（计算量小）
    - 作为更复杂网络的baseline

优点：
    ✓ 计算快速（无隐藏层）
    ✓ 参数少（只有一层权重）
    ✓ 易于理解和调试
    ✓ 仍然具有自适应能力

缺点：
    ✗ 表达能力有限（无隐藏层）
    ✗ 无长期记忆（无循环连接）
    ✗ 不适合复杂任务
"""

import torch
import numpy as np

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs


class AdaptiveLinearNet:
    """
    自适应线性神经网络
    
    一个简化版的自适应网络，只包含输入层和输出层，没有隐藏层。
    权重可以根据神经元的激活状态动态更新，实现快速适应和短期记忆。
    
    只需要一个CPPN节点：
    - delta_w_node: 生成权重更新规则
    
    权重更新公式：
        output(t) = activation(W(t) * input(t))
        delta_w = CPPN(坐标, input(t), output(t), W(t))
        W(t+1) = W(t) + delta_w（仅在已表达的连接上）
        
    注意：权重只在初始时非零的连接上更新，保持网络稀疏性。
    """
    def __init__(
        self,
        delta_w_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):
        """
        初始化自适应线性网络
        
        Args:
            delta_w_node: CPPN节点，生成权重更新量（这是唯一的CPPN）
            input_coords (list): 输入神经元的2D坐标列表
            output_coords (list): 输出神经元的2D坐标列表
            weight_threshold (float): 权重阈值，小于此值的初始权重被设为0
            weight_max (float): 权重最大值，权重被限制在[-weight_max, weight_max]
            activation (callable): 输出层激活函数，默认tanh
            cppn_activation (callable): CPPN输出的激活函数，默认identity
                                       对CPPN输出应用激活可以控制更新的范围和形式
            batch_size (int): 批量大小
            device (str): 计算设备
        """

        # 保存CPPN节点（唯一的CPPN，负责生成权重更新）
        self.delta_w_node = delta_w_node

        # 保存网络结构信息
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(
            input_coords, dtype=torch.float32, device=device
        )

        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(
            output_coords, dtype=torch.float32, device=device
        )

        # 保存超参数
        self.weight_threshold = weight_threshold  # 初始权重稀疏化阈值
        self.weight_max = weight_max  # 权重最大值（防止爆炸）

        # 保存激活函数
        self.activation = activation  # 输出层激活函数
        self.cppn_activation = cppn_activation  # CPPN输出激活函数

        self.batch_size = batch_size
        self.device = device
        
        # 初始化网络权重
        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        """
        使用CPPN生成初始权重矩阵
        
        与AdaptiveNet不同，这里直接使用delta_w_node来生成初始权重。
        这意味着初始权重就是"从零权重开始的第一次更新"。
        
        Args:
            in_coords (torch.Tensor): 输入神经元坐标
            out_coords (torch.Tensor): 输出神经元坐标
            w_node: CPPN节点（实际上是delta_w_node）
            
        Returns:
            torch.Tensor: 初始权重矩阵，形状(n_out, n_in)
        """
        # 生成坐标输入
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

        n_in = len(in_coords)
        n_out = len(out_coords)

        # 创建零张量作为初始激活值和权重
        zeros = torch.zeros((n_out, n_in), dtype=torch.float32, device=self.device)

        # 调用CPPN生成初始权重
        # 注意：这里对CPPN输出应用了cppn_activation
        # 这可以控制初始权重的范围和分布
        weights = self.cppn_activation(
            w_node(
                x_out=x_out,
                y_out=y_out,
                x_in=x_in,
                y_in=y_in,
                pre=zeros,  # 初始输入激活为0
                post=zeros,  # 初始输出激活为0
                w=zeros,  # 初始权重为0
            )
        )
        
        # 对权重进行阈值化和限幅
        clamp_weights_(weights, self.weight_threshold, self.weight_max)

        return weights

    def reset(self):
        """
        重置网络：重新生成初始权重
        
        生成输入到输出的权重矩阵，并记录哪些连接是非零的。
        只有初始时非零的连接才会在后续步骤中被更新，这保持了网络的稀疏性。
        
        生成的数据：
        - input_to_output: 权重矩阵，形状(batch_size, n_outputs, n_inputs)
                          每个batch可以有独立的权重（独立适应）
        - w_expressed: 布尔掩码，标记哪些连接是激活的（非零）
        - batched_coords: 预计算的批量坐标，用于权重更新
        """
        with torch.no_grad():
            # 生成初始权重矩阵
            base_weights = self.get_init_weights(
                self.input_coords, self.output_coords, self.delta_w_node
            )
            self.input_to_output = base_weights.unsqueeze(0).repeat(
                self.batch_size, 1, 1
            ).contiguous()

            # 记录哪些连接是表达的（非零）
            # 这个掩码用于确保只有初始非零的连接才会被更新
            # 保持网络稀疏性，避免密集化
            self.w_expressed = self.input_to_output != 0

            # 预计算批量坐标（用于后续的权重更新）
            self.batched_coords = get_coord_inputs(
                self.input_coords, self.output_coords, batch_size=self.batch_size
            )

    def activate(self, inputs):
        """
        前向传播并更新自适应权重
        
        执行三个步骤：
        1. 使用当前权重计算输出
        2. 使用CPPN根据输入、输出和当前权重计算权重更新量
        3. 更新权重（仅在已表达的连接上）
        
        自适应更新公式：
            output(t) = activation(W(t) * input(t))
            delta_w = CPPN(坐标, input(t), output(t), W(t))
            W(t+1)[expressed] = W(t)[expressed] + delta_w[expressed]
        
        关键特性：
        - 权重只在初始非零的连接上更新（保持稀疏性）
        - 每个batch独立更新权重（独立适应）
        - 权重被限制在[-weight_max, weight_max]范围内
        
        这使网络能够：
        - 根据输入输出的相关性调整连接强度（类似Hebbian学习）
        - 实现短期记忆（权重变化保留信息）
        - 快速适应环境变化
        
        Args:
            inputs: 输入数据，形状为 (batch_size, n_inputs)
                   可以是numpy数组、列表或torch张量
                   
        Returns:
            torch.Tensor: 输出数据，形状为 (batch_size, n_outputs)
        """
        with torch.no_grad():
            # 将输入转换为张量
            # 如果输入是列表（如多个numpy数组），先合并成单个数组
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            # 转换为torch张量并添加维度以适配矩阵乘法
            # 形状: (batch_size, n_inputs) -> (batch_size, n_inputs, 1)
            inputs = torch.tensor(
                inputs, dtype=torch.float32, device=self.device
            ).unsqueeze(2)

            # 步骤1：计算输出（使用当前权重）
            # 矩阵乘法: (batch_size, n_outputs, n_inputs) @ (batch_size, n_inputs, 1)
            #         = (batch_size, n_outputs, 1)
            outputs = self.activation(self.input_to_output.matmul(inputs))

            # 步骤2：准备权重更新所需的数据
            # 将输入激活值扩展成矩阵形式（每个连接都能看到输入激活）
            input_activs = inputs.transpose(1, 2).expand(
                self.batch_size, self.n_outputs, self.n_inputs
            )
            # 将输出激活值扩展成矩阵形式（每个连接都能看到输出激活）
            output_activs = outputs.expand(
                self.batch_size, self.n_outputs, self.n_inputs
            )

            # 获取预计算的坐标
            (x_out, y_out), (x_in, y_in) = self.batched_coords

            # 步骤3：使用CPPN计算权重更新量
            # CPPN输入：坐标 + 输入激活 + 输出激活 + 当前权重
            # CPPN输出：权重更新量 delta_w
            delta_w = self.cppn_activation(
                self.delta_w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=input_activs,  # 输入激活（pre-synaptic）
                    post=output_activs,  # 输出激活（post-synaptic）
                    w=self.input_to_output,  # 当前权重
                )
            )

            # 保存delta_w（可用于调试和分析）
            self.delta_w = delta_w

            # 步骤4：应用权重更新（仅在已表达的连接上）
            # w_expressed是一个布尔掩码，标记哪些连接是激活的
            # 这保持了网络的稀疏性，防止网络变得全连接
            self.input_to_output[self.w_expressed] += delta_w[self.w_expressed]
            
            # 限制权重在合理范围内（防止爆炸或过小）
            # weight_threshold=0.0表示不进行额外的阈值化（已在初始化时完成）
            clamp_weights_(
                self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max
            )

        # 移除额外维度并返回
        # 形状: (batch_size, n_outputs, 1) -> (batch_size, n_outputs)
        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
        device="cuda:0",
    ):
        """
        从NEAT基因组创建自适应线性网络（工厂方法）
        
        这是创建AdaptiveLinearNet的标准方法。与AdaptiveNet相比，
        这个网络更简单，只需要一个CPPN节点。
        
        工作流程：
        1. 使用create_cppn从基因组创建1个CPPN节点
        2. 这个CPPN节点负责生成权重更新规则
        3. 使用这个CPPN节点和坐标信息创建AdaptiveLinearNet实例
        
        CPPN的输入：
        - x_in, y_in: 输入神经元的坐标
        - x_out, y_out: 输出神经元的坐标
        - pre: 输入神经元的激活值
        - post: 输出神经元的激活值
        - w: 当前权重值
        
        CPPN的输出：
        - delta_w: 权重更新量
        
        Args:
            genome: NEAT基因组，由NEAT算法进化得到
            config: NEAT配置对象
            input_coords (list): 输入神经元的2D坐标
                                例如：[[-1, 0], [0, 0], [1, 0], [0, -1]]
                                可以根据任务设计合理的空间布局
            output_coords (list): 输出神经元的2D坐标
                                 例如：[[-1, 0], [0, 0], [1, 0]]
            weight_threshold (float): 初始权重阈值，控制网络稀疏性
            weight_max (float): 权重最大值，防止权重爆炸
            output_activation (callable, optional): 覆盖CPPN输出节点的激活函数
            activation (callable): 输出层激活函数，默认tanh
            cppn_activation (callable): CPPN输出的激活函数，默认identity
                                       可以设为tanh来限制权重更新的幅度
            batch_size (int): 批量大小
            device (str): 计算设备
            
        Returns:
            AdaptiveLinearNet: 创建的自适应线性网络实例
        """
        # 从基因组创建1个CPPN节点
        # CPPN接收7个输入（坐标、激活、权重），产生1个输出（权重更新）
        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],  # CPPN输入
            ["delta_w"],  # CPPN输出（权重更新规则）
            output_activation=output_activation,  # 可选的输出激活函数
        )

        # 提取CPPN节点
        delta_w_node = nodes[0]

        # 创建并返回AdaptiveLinearNet实例
        return AdaptiveLinearNet(
            delta_w_node,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
            device=device,
        )
