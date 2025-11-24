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
递归神经网络模块 (RecurrentNet)

这是最简单和最快的网络类型，实现了支持循环连接的递归神经网络。
它不使用CPPN，权重直接从NEAT基因组中提取，因此是标准的NEAT实现。

网络结构：
    输入层 --> 隐藏层 ↺ --> 输出层 ↺
               ↓  ↑         ↓  ↑
              循环连接      循环连接

关键特性：
    - 固定权重：权重在初始化后不再改变（由NEAT进化）
    - 循环连接：支持隐藏层和输出层的自循环
    - 可配置：支持多次内部迭代，增强计算能力
    - 高效：使用密集矩阵，批量计算快速

与其他网络的区别：
    - 不使用CPPN，权重直接编码在基因组中
    - 不使用空间坐标，网络拓扑完全由NEAT进化
    - 权重固定，无自适应机制
    - 计算最快，内存占用最小

适用场景：
    - 简单的控制任务（CartPole、Pendulum等）
    - 不需要在线学习的任务
    - 需要快速评估的场景
    - 作为baseline比较

优点：
    ✓ 实现简单，易于理解
    ✓ 计算速度最快
    ✓ 内存占用最小
    ✓ 训练稳定（无权重更新）
    ✓ 是标准的NEAT实现

缺点：
    ✗ 无自适应能力
    ✗ 无在线学习
    ✗ 权重完全进化，搜索空间大
"""

import torch
import numpy as np
from .activations import sigmoid_activation


# 稀疏矩阵版本（已弃用，保留作为参考）
# def sparse_mat(shape, conns):
#     idxs, weights = conns
#     if len(idxs) > 0:
#         idxs = torch.LongTensor(idxs).t()
#         weights = torch.FloatTensor(weights)
#         mat = torch.sparse.FloatTensor(idxs, weights, shape)
#     else:
#         mat = torch.sparse.FloatTensor(shape[0], shape[1])
#     return mat


def dense_from_coo(shape, conns, dtype=torch.float64, device="cpu"):
    """
    从COO格式创建密集矩阵
    
    COO (COOrdinate) 格式是一种稀疏矩阵表示方法，只存储非零元素。
    这个函数将COO格式转换为密集的PyTorch张量。
    
    Args:
        shape (tuple): 矩阵形状，如 (n_rows, n_cols)
        conns (tuple): 连接信息，格式为 (indices, weights)
                      - indices: [(row_0, col_0), (row_1, col_1), ...]
                      - weights: [weight_0, weight_1, ...]
        dtype: PyTorch数据类型，默认float64
        
    Returns:
        torch.Tensor: 密集矩阵，大部分元素为0
        
    示例：
        shape = (3, 4)
        indices = [(0, 1), (1, 2), (2, 0)]
        weights = [0.5, -0.3, 0.8]
        mat = dense_from_coo(shape, (indices, weights))
        # mat = [[0.0, 0.5, 0.0, 0.0],
        #        [0.0, 0.0, -0.3, 0.0],
        #        [0.8, 0.0, 0.0, 0.0]]
    """
    # 创建全零矩阵
    mat = torch.zeros(shape, dtype=dtype, device=device)
    
    # 解包连接信息
    idxs, weights = conns
    
    # 如果没有连接，直接返回零矩阵
    if len(idxs) == 0:
        return mat
    
    # 将索引列表转换为行列索引
    # [(r0,c0), (r1,c1), ...] -> rows=[r0,r1,...], cols=[c0,c1,...]
    rows, cols = np.array(idxs).transpose()
    
    # 填充非零元素
    mat[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = torch.tensor(
        weights, dtype=dtype, device=device)
    
    return mat


class RecurrentNet():
    """
    递归神经网络
    
    支持任意拓扑的递归神经网络，包括隐藏层和输出层的自循环。
    权重直接从NEAT基因组中提取，不使用CPPN或自适应机制。
    
    网络可以包含多种连接：
    - 输入 -> 隐藏
    - 隐藏 -> 隐藏（循环）
    - 输出 -> 隐藏（反馈）
    - 输入 -> 输出
    - 隐藏 -> 输出
    - 输出 -> 输出（循环）
    
    每个时间步的计算：
        1. 更新隐藏层（内部迭代n_internal_steps次）：
           hidden(t) = activation(response * (
                W_ih * input(t) +
                W_hh * hidden(t-1) +
                W_oh * output(t-1) + bias_h))
        
        2. 更新输出层：
           output(t) = activation(response * (
                W_io * input(t) +
                W_ho * hidden(t) +
                W_oo * output(t-1) + bias_o))
    """
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 input_to_hidden, hidden_to_hidden, output_to_hidden,
                 input_to_output, hidden_to_output, output_to_output,
                 hidden_responses, output_responses,
                 hidden_biases, output_biases,
                 batch_size=1,
                 use_current_activs=False,
                 activation=sigmoid_activation,
                 n_internal_steps=1,
                 dtype=torch.float64,
                 device="cpu"):
        """
        初始化递归神经网络
        
        Args:
            n_inputs (int): 输入节点数量
            n_hidden (int): 隐藏节点数量（可以为0）
            n_outputs (int): 输出节点数量
            input_to_hidden (tuple): 输入到隐藏层的连接 (indices, weights)
            hidden_to_hidden (tuple): 隐藏层循环连接 (indices, weights)
            output_to_hidden (tuple): 输出到隐藏层的反馈连接 (indices, weights)
            input_to_output (tuple): 输入到输出层的连接 (indices, weights)
            hidden_to_output (tuple): 隐藏到输出层的连接 (indices, weights)
            output_to_output (tuple): 输出层循环连接 (indices, weights)
            hidden_responses (list): 隐藏层节点的响应系数
            output_responses (list): 输出层节点的响应系数
            hidden_biases (list): 隐藏层节点的偏置
            output_biases (list): 输出层节点的偏置
            batch_size (int): 批量大小
            use_current_activs (bool): 计算输出时是否使用当前时刻的隐藏层激活
                                      False: 使用上一时刻的隐藏层激活
                                      True: 使用当前时刻的隐藏层激活
            activation (callable): 激活函数，默认sigmoid
            n_internal_steps (int): 隐藏层的内部迭代次数，增加可提升计算能力
            dtype: PyTorch数据类型，默认float64
        """

        # 保存配置参数
        self.use_current_activs = use_current_activs  # 是否使用当前隐藏层激活
        self.activation = activation  # 激活函数
        self.n_internal_steps = n_internal_steps  # 内部迭代次数
        self.dtype = dtype  # 数据类型
        self.device = torch.device(device)

        # 保存网络结构信息
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # 创建权重矩阵（仅当有隐藏层时）
        if n_hidden > 0:
            # 输入 -> 隐藏：(n_hidden, n_inputs)
            self.input_to_hidden = dense_from_coo(
                (n_hidden, n_inputs), input_to_hidden, dtype=dtype, device=self.device)
            # 隐藏 -> 隐藏（循环）：(n_hidden, n_hidden)
            self.hidden_to_hidden = dense_from_coo(
                (n_hidden, n_hidden), hidden_to_hidden, dtype=dtype, device=self.device)
            # 输出 -> 隐藏（反馈）：(n_hidden, n_outputs)
            self.output_to_hidden = dense_from_coo(
                (n_hidden, n_outputs), output_to_hidden, dtype=dtype, device=self.device)
            # 隐藏 -> 输出：(n_outputs, n_hidden)
            self.hidden_to_output = dense_from_coo(
                (n_outputs, n_hidden), hidden_to_output, dtype=dtype, device=self.device)
        
        # 输入 -> 输出：(n_outputs, n_inputs)
        self.input_to_output = dense_from_coo(
            (n_outputs, n_inputs), input_to_output, dtype=dtype, device=self.device)
        # 输出 -> 输出（循环）：(n_outputs, n_outputs)
        self.output_to_output = dense_from_coo(
            (n_outputs, n_outputs), output_to_output, dtype=dtype, device=self.device)

        # 创建响应系数和偏置张量（仅当有隐藏层时）
        if n_hidden > 0:
            # 隐藏层的响应系数（缩放因子）
            self.hidden_responses = torch.tensor(hidden_responses, dtype=dtype, device=self.device)
            # 隐藏层的偏置
            self.hidden_biases = torch.tensor(hidden_biases, dtype=dtype, device=self.device)

        # 输出层的响应系数
        self.output_responses = torch.tensor(
            output_responses, dtype=dtype, device=self.device)
        # 输出层的偏置
        self.output_biases = torch.tensor(output_biases, dtype=dtype, device=self.device)

        # 初始化网络状态
        self.reset(batch_size)

    def reset(self, batch_size=1):
        """
        重置网络状态（清空记忆）
        
        将隐藏层和输出层的激活值重置为零。
        这应该在每个episode开始时调用，清空网络的记忆。
        
        Args:
            batch_size (int): 批量大小，可以与初始化时不同
        """
        # 重置隐藏层激活值（如果有隐藏层）
        if self.n_hidden > 0:
            self.activs = torch.zeros(
                batch_size, self.n_hidden, dtype=self.dtype, device=self.device)
        else:
            self.activs = None  # 无隐藏层
        
        # 重置输出层激活值
        self.outputs = torch.zeros(
            batch_size, self.n_outputs, dtype=self.dtype, device=self.device)

    def activate(self, inputs):
        """
        前向传播：计算网络输出
        
        执行一个时间步的计算：
        1. 更新隐藏层（如果有），进行n_internal_steps次内部迭代
        2. 更新输出层
        3. 保存新的隐藏层和输出层状态（用于下一时间步）
        
        隐藏层更新公式（迭代n_internal_steps次）：
            hidden(t) = activation(response * (
                W_ih * input(t) +
                W_hh * hidden(t-1) +
                W_oh * output(t-1)) + bias)
        
        输出层更新公式：
            output(t) = activation(response * (
                W_io * input(t) +
                W_ho * hidden(t) +
                W_oo * output(t-1)) + bias)
        
        注意：内部迭代可以增强网络的计算能力，使其能够进行更复杂的推理。
        
        Args:
            inputs: 输入数据，形状为 (batch_size, n_inputs)
                   可以是numpy数组、列表或torch张量
                   
        Returns:
            torch.Tensor: 输出数据，形状为 (batch_size, n_outputs)
        """
        with torch.no_grad():
            # 将输入转换为张量
            # 如果输入是列表（如多个numpy数组），先合并
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=self.dtype, device=self.device)
            else:
                inputs = inputs.to(self.device, dtype=self.dtype)
            
            # 默认使用上一时刻的隐藏层激活（用于计算输出）
            activs_for_output = self.activs
            
            # 步骤1：更新隐藏层（如果有）
            if self.n_hidden > 0:
                # 进行n_internal_steps次内部迭代
                # 这可以让网络在单个时间步内进行多次"思考"
                for _ in range(self.n_internal_steps):
                    # 计算隐藏层的新激活值
                    # 三个输入来源：
                    # 1) 当前输入
                    # 2) 上一时刻的隐藏层（循环连接，提供记忆）
                    # 3) 上一时刻的输出（反馈连接）
                    self.activs = self.activation(self.hidden_responses * (
                        self.input_to_hidden.mm(inputs.t()).t() +  # 输入贡献
                        self.hidden_to_hidden.mm(self.activs.t()).t() +  # 隐藏层循环
                        self.output_to_hidden.mm(self.outputs.t()).t()) +  # 输出反馈
                        self.hidden_biases)  # 偏置
                
                # 如果设置了use_current_activs，使用当前时刻的隐藏层激活
                if self.use_current_activs:
                    activs_for_output = self.activs
            
            # 步骤2：计算输出层的输入
            # 两个主要来源：
            # 1) 当前输入的直接贡献
            # 2) 上一时刻输出的循环贡献
            output_inputs = (self.input_to_output.mm(inputs.t()).t() +
                             self.output_to_output.mm(self.outputs.t()).t())
            
            # 添加隐藏层的贡献（如果有）
            if self.n_hidden > 0:
                output_inputs += self.hidden_to_output.mm(
                    activs_for_output.t()).t()
            
            # 步骤3：更新输出层
            self.outputs = self.activation(
                self.output_responses * output_inputs + self.output_biases)
        
        # 返回输出（同时也保存在self.outputs中，用于下一时间步）
        return self.outputs

    @staticmethod
    def create(genome, config, batch_size=1, activation=sigmoid_activation,
               prune_empty=False, use_current_activs=False, n_internal_steps=1,
               dtype=torch.float64, device="cpu"):
        """
        从NEAT基因组创建递归神经网络（工厂方法）
        
        这是创建RecurrentNet的标准方法。它从NEAT基因组中提取所有连接和节点信息，
        构建完整的递归神经网络。与自适应网络不同，这里不使用CPPN，
        权重直接从基因组中读取。
        
        工作流程：
        1. 分析基因组，找出对输出必需的节点和连接
        2. 将节点分类为输入、隐藏、输出三类
        3. 将连接分类为6种类型（输入->隐藏、隐藏->隐藏等）
        4. 提取每个节点的响应系数和偏置
        5. 创建RecurrentNet实例
        
        Args:
            genome: NEAT基因组，包含完整的网络拓扑和权重
            config: NEAT配置对象
            batch_size (int): 批量大小
            activation (callable): 激活函数，默认sigmoid
            prune_empty (bool): 是否剪枝空节点（没有输入的节点）
                               这可以减小网络大小，提高效率
            use_current_activs (bool): 计算输出时是否使用当前隐藏层激活
            n_internal_steps (int): 隐藏层内部迭代次数
                                   >1可以增强计算能力，但会变慢
                                   
        Returns:
            RecurrentNet: 创建的递归神经网络实例
            
        使用示例：
            # 简单使用（默认参数）
            net = RecurrentNet.create(genome, config)
            outputs = net.activate(inputs)
            
            # 增强计算能力
            net = RecurrentNet.create(
                genome, config,
                batch_size=4,
                n_internal_steps=3,  # 更多内部迭代
                activation=tanh_activation
            )
            
            # 每个episode开始时重置
            net.reset(batch_size)
            for step in range(max_steps):
                outputs = net.activate(states)
                # ... 使用outputs ...
        """
        from neat.graphs import required_for_output

        genome_config = config.genome_config
        
        # 找出对生成输出必需的所有节点（剪枝不需要的节点）
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections)
        
        # 如果启用剪枝，找出所有非空节点（有输入的节点）
        if prune_empty:
            # 收集所有激活连接的目标节点，加上输入节点
            nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))

        # 将所有节点分类为输入、隐藏、输出三类
        input_keys = list(genome_config.input_keys)  # 输入节点ID列表
        hidden_keys = [k for k in genome.nodes.keys()
                       if k not in genome_config.output_keys]  # 隐藏节点ID列表
        output_keys = list(genome_config.output_keys)  # 输出节点ID列表

        # 提取隐藏层节点的响应系数（缩放因子）
        hidden_responses = [genome.nodes[k].response for k in hidden_keys]
        # 提取输出层节点的响应系数
        output_responses = [genome.nodes[k].response for k in output_keys]

        # 提取隐藏层节点的偏置
        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        # 提取输出层节点的偏置
        output_biases = [genome.nodes[k].bias for k in output_keys]

        # 如果启用剪枝，将空输出节点的偏置设为0
        # 这样即使节点没有输入，也不会产生非零输出
        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    output_biases[i] = 0.0

        # 计算各层的节点数量
        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)

        # 创建节点ID到索引的映射（用于构建连接矩阵）
        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}

        def key_to_idx(key):
            """将节点ID转换为对应层内的索引"""
            if key in input_keys:
                return input_key_to_idx[key]
            elif key in hidden_keys:
                return hidden_key_to_idx[key]
            elif key in output_keys:
                return output_key_to_idx[key]

        # 初始化6种连接类型的列表
        # 每种连接用 (indices, weights) 表示，其中 indices = [(row, col), ...]
        input_to_hidden = ([], [])  # 输入 -> 隐藏
        hidden_to_hidden = ([], [])  # 隐藏 -> 隐藏（循环）
        output_to_hidden = ([], [])  # 输出 -> 隐藏（反馈）
        input_to_output = ([], [])  # 输入 -> 输出
        hidden_to_output = ([], [])  # 隐藏 -> 输出
        output_to_output = ([], [])  # 输出 -> 输出（循环）

        # 遍历所有连接，根据源节点和目标节点的类型分类
        for conn in genome.connections.values():
            # 跳过未激活的连接
            if not conn.enabled:
                continue

            # 获取连接的源节点和目标节点
            i_key, o_key = conn.key
            
            # 跳过不必需的连接（不影响输出）
            if o_key not in required and i_key not in required:
                continue
            
            # 如果启用剪枝，跳过源自空节点的连接
            if prune_empty and i_key not in nonempty:
                print('Pruned {}'.format(conn.key))
                continue

            # 将节点ID转换为索引
            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)

            # 根据连接的源和目标节点类型，将连接添加到相应列表
            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = input_to_hidden
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hidden_to_hidden
            elif i_key in output_keys and o_key in hidden_keys:
                idxs, vals = output_to_hidden
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = input_to_output
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hidden_to_output
            elif i_key in output_keys and o_key in output_keys:
                idxs, vals = output_to_output
            else:
                # 不应该到达这里（如果基因组有效）
                raise ValueError(
                    'Invalid connection from key {} to key {}'.format(i_key, o_key))

            # 添加连接索引和权重
            # 注意：索引顺序是 (目标, 源)，因为矩阵乘法是 W @ x
            idxs.append((o_idx, i_idx))  # (to, from)
            vals.append(conn.weight)  # 连接权重

        # 创建并返回RecurrentNet实例
        return RecurrentNet(n_inputs, n_hidden, n_outputs,
                            input_to_hidden, hidden_to_hidden, output_to_hidden,
                            input_to_output, hidden_to_output, output_to_output,
                            hidden_responses, output_responses,
                            hidden_biases, output_biases,
                            batch_size=batch_size,
                            activation=activation,
                            use_current_activs=use_current_activs,
                            n_internal_steps=n_internal_steps,
                            dtype=dtype,
                            device=device)
