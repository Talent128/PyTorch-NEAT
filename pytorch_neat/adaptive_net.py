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
自适应递归网络模块 (AdaptiveNet)

这是最复杂和最强大的网络类型，结合了：
1. 递归结构：包含隐藏层和循环连接，支持长期记忆
2. 自适应权重：隐藏层到隐藏层的权重可以动态更新
3. CPPN生成：使用多个CPPN分别生成不同的权重矩阵和更新规则

网络结构：
    输入层 --[w_ih]--> 隐藏层 ↺[w_hh, 自适应] --> 输出层
                        ↓                        ↑
                      [b_h]                   [w_ho]
                                               [b_o]

自适应机制：
    - 初始化时，所有权重由对应的CPPN根据神经元坐标生成
    - 每个时间步，隐藏层到隐藏层的权重根据神经元激活值更新：
      w_hh(t+1) = w_hh(t) + delta_w_node(坐标, 激活值, 当前权重)
    - 这使网络能够根据经验动态调整，实现快速适应和元学习

适用场景：
    - 需要复杂记忆和适应能力的任务
    - 非平稳环境（环境规则会变化）
    - 需要在线学习的任务
    - 元学习和少样本学习

与其他网络的比较：
    - RecurrentNet: 固定权重的递归网络，计算快但适应能力弱
    - AdaptiveLinearNet: 无隐藏层的自适应网络，简单但表达能力有限
    - AdaptiveNet: 最强大但计算最慢，适合复杂任务
"""

import torch
from .activations import tanh_activation
from .cppn import create_cppn, clamp_weights_, get_coord_inputs


class AdaptiveNet:
    """
    自适应递归神经网络
    
    这个网络包含输入层、隐藏层和输出层，其中隐藏层到隐藏层的权重
    可以根据神经元的激活状态动态更新。
    
    网络有6个CPPN节点，分别负责生成/更新不同的权重矩阵：
    1. w_ih_node: 生成输入到隐藏层的权重矩阵
    2. b_h_node: 生成隐藏层的偏置
    3. w_hh_node: 生成隐藏层到隐藏层的初始权重矩阵
    4. b_o_node: 生成输出层的偏置
    5. w_ho_node: 生成隐藏层到输出层的权重矩阵
    6. delta_w_node: 生成隐藏层到隐藏层权重的更新量（自适应）
    
    工作流程：
        1. 初始化时，使用前5个CPPN生成所有初始权重
        2. 前向传播时，计算隐藏层和输出层的激活值
        3. 使用delta_w_node根据当前激活值更新w_hh权重
        4. 更新后的权重在下一个时间步生效
    """
    def __init__(self,

                 w_ih_node,
                 b_h_node,
                 w_hh_node,
                 b_o_node,
                 w_ho_node,
                 delta_w_node,
                 #  stateful_node,

                 input_coords,
                 hidden_coords,
                 output_coords,

                 weight_threshold=0.2,
                 activation=tanh_activation,

                 batch_size=1,
                 device='cuda:0'):
        """
        初始化自适应递归网络
        
        Args:
            w_ih_node: CPPN节点，生成输入到隐藏层的权重
            b_h_node: CPPN节点，生成隐藏层的偏置
            w_hh_node: CPPN节点，生成隐藏层循环权重的初始值
            b_o_node: CPPN节点，生成输出层的偏置
            w_ho_node: CPPN节点，生成隐藏层到输出层的权重
            delta_w_node: CPPN节点，生成隐藏层循环权重的更新量（自适应核心）
            input_coords (list): 输入神经元的2D坐标列表，如[[-1, 0], [0, 0], [1, 0]]
            hidden_coords (list): 隐藏神经元的2D坐标列表
            output_coords (list): 输出神经元的2D坐标列表
            weight_threshold (float): 权重阈值，小于此值的权重被设为0
            activation (callable): 激活函数，默认tanh
            batch_size (int): 批量大小
            device (str): 计算设备，'cuda:0'或'cpu'
        """

        # 保存所有CPPN节点
        self.w_ih_node = w_ih_node  # 输入->隐藏权重生成器

        self.b_h_node = b_h_node  # 隐藏层偏置生成器
        self.w_hh_node = w_hh_node  # 隐藏->隐藏初始权重生成器

        self.b_o_node = b_o_node  # 输出层偏置生成器
        self.w_ho_node = w_ho_node  # 隐藏->输出权重生成器

        self.delta_w_node = delta_w_node  # 隐藏->隐藏权重更新生成器（自适应核心）
        # self.stateful_node = stateful_node  # 可选的状态节点（未使用）

        # 保存网络结构信息
        self.n_inputs = len(input_coords)
        self.input_coords = torch.tensor(
            input_coords, dtype=torch.float32, device=device)

        self.n_hidden = len(hidden_coords)
        self.hidden_coords = torch.tensor(
            hidden_coords, dtype=torch.float32, device=device)

        self.n_outputs = len(output_coords)
        self.output_coords = torch.tensor(
            output_coords, dtype=torch.float32, device=device)

        # 保存超参数
        self.weight_threshold = weight_threshold  # 权重稀疏化阈值

        self.activation = activation  # 激活函数

        self.batch_size = batch_size
        self.device = device
        
        # 初始化网络权重和状态
        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        """
        使用CPPN生成初始权重矩阵
        
        根据输入和输出神经元的空间坐标，使用CPPN生成连接权重。
        这是HyperNEAT的核心思想：利用几何规律性间接编码权重。
        
        Args:
            in_coords (torch.Tensor): 输入神经元坐标，形状(n_in, 2)
            out_coords (torch.Tensor): 输出神经元坐标，形状(n_out, 2)
            w_node: CPPN节点，接收坐标和激活值，输出权重
            
        Returns:
            torch.Tensor: 权重矩阵，形状(n_out, n_in)
                         其中weights[i, j]是从输入j到输出i的权重
        """
        # 生成坐标输入：为每对神经元创建坐标
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

        n_in = len(in_coords)
        n_out = len(out_coords)

        # 创建零张量作为初始激活值和权重（CPPN也需要这些输入）
        zeros = torch.zeros(
            (n_out, n_in), dtype=torch.float32, device=self.device)

        # 调用CPPN生成权重
        # 输入：神经元坐标 + 零激活值 + 零权重
        # 输出：初始权重矩阵
        weights = w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in,
                         pre=zeros, post=zeros, w=zeros)
        
        # 对权重进行阈值化和限幅处理（稀疏化）
        clamp_weights_(weights, self.weight_threshold)

        return weights

    def reset(self):
        """
        重置网络：重新生成所有权重矩阵并清空状态
        
        使用各个CPPN节点根据神经元坐标生成初始权重矩阵。
        这个方法在创建网络时调用一次，也可以在需要时重置网络。
        
        生成的权重矩阵：
        - input_to_hidden: 输入层到隐藏层的权重，形状(n_hidden, n_inputs)
        - bias_hidden: 隐藏层的偏置，形状(batch_size, n_hidden, 1)
        - hidden_to_hidden: 隐藏层循环权重，形状(batch_size, n_hidden, n_hidden)
                           注意：这个权重会在每个时间步动态更新！
        - bias_output: 输出层的偏置，形状(n_outputs, 1)
        - hidden_to_output: 隐藏层到输出层的权重，形状(n_outputs, n_hidden)
        - hidden: 隐藏层状态，初始化为零，形状(batch_size, n_hidden, 1)
        """
        with torch.no_grad():
            # 生成输入到隐藏层的权重
            self.input_to_hidden = self.get_init_weights(
                self.input_coords, self.hidden_coords, self.w_ih_node)

            # 生成隐藏层偏置
            # 使用虚拟坐标(0, 0)来生成偏置（偏置可以看作从常数1的连接）
            bias_coords = torch.zeros(
                (1, 2), dtype=torch.float32, device=self.device)
            self.bias_hidden = self.get_init_weights(
                bias_coords, self.hidden_coords, self.b_h_node).unsqueeze(0).expand(
                    self.batch_size, self.n_hidden, 1)

            # 生成隐藏层到隐藏层的初始权重（循环连接）
            # 扩展到批量维度（每个样本有独立的权重矩阵，可以独立更新）
            base_hidden_to_hidden = self.get_init_weights(
                self.hidden_coords, self.hidden_coords, self.w_hh_node
            )
            self.hidden_to_hidden = base_hidden_to_hidden.unsqueeze(0).repeat(
                self.batch_size, 1, 1
            ).contiguous()

            # 生成输出层偏置
            bias_coords = torch.zeros(
                (1, 2), dtype=torch.float32, device=self.device)
            self.bias_output = self.get_init_weights(
                bias_coords, self.output_coords, self.b_o_node)

            # 生成隐藏层到输出层的权重
            self.hidden_to_output = self.get_init_weights(
                self.hidden_coords, self.output_coords, self.w_ho_node)

            # 初始化隐藏层状态为零（记忆清空）
            self.hidden = torch.zeros((self.batch_size, self.n_hidden, 1),
                                      dtype=torch.float32,
                                      device=self.device)

            # 预计算批量坐标（用于权重更新时的CPPN输入）
            self.batched_hidden_coords = get_coord_inputs(
                self.hidden_coords, self.hidden_coords, batch_size=self.batch_size)
            
            # 可选的CPPN状态（未使用）
            # self.cppn_state = torch.zeros(
            #     (self.batch_size, self.n_hidden, self.n_hidden))

    def activate(self, inputs):
        """
        前向传播并更新自适应权重
        
        这是网络的核心方法，执行三个步骤：
        1. 计算隐藏层激活值（使用当前权重）
        2. 计算输出层激活值
        3. 根据当前激活值更新隐藏层循环权重（自适应）
        
        自适应更新公式：
            hidden(t) = activation(W_ih * input(t) + W_hh(t) * hidden(t-1) + b_h)
            output(t) = activation(W_ho * hidden(t) + b_o)
            W_hh(t+1) = W_hh(t) + delta_w_node(coords, hidden(t), W_hh(t))
        
        这使网络能够：
        - 根据经验调整连接强度
        - 实现快速适应（meta-learning）
        - 在单个episode内学习
        
        Args:
            inputs: 输入数据，形状为 (batch_size, n_inputs)
                   可以是numpy数组或列表
                   
        Returns:
            torch.Tensor: 输出数据，形状为 (batch_size, n_outputs)
        """
        with torch.no_grad():
            # 转换输入为张量，并添加维度以适配矩阵乘法
            # 形状: (batch_size, n_inputs) -> (batch_size, n_inputs, 1)
            inputs = torch.tensor(
                inputs, dtype=torch.float32, device=self.device).unsqueeze(2)

            # 步骤1：更新隐藏层状态（递归神经网络的核心）
            # 计算三部分的贡献：
            # 1) input_to_hidden.matmul(inputs): 当前输入的贡献
            # 2) hidden_to_hidden.matmul(self.hidden): 上一时刻隐藏层的贡献（记忆）
            # 3) bias_hidden: 偏置项
            self.hidden = self.activation(self.input_to_hidden.matmul(inputs) +
                                          self.hidden_to_hidden.matmul(self.hidden) +
                                          self.bias_hidden)

            # 步骤2：计算输出层激活值
            outputs = self.activation(
                self.hidden_to_output.matmul(self.hidden) +
                self.bias_output)

            # 步骤3：准备权重更新所需的数据
            # 将隐藏层激活值扩展成矩阵，以便为每个连接提供激活值
            # hidden_outputs[i, j, k] = hidden[i, j] (post-synaptic)
            hidden_outputs = self.hidden.expand(
                self.batch_size, self.n_hidden, self.n_hidden)
            # hidden_inputs[i, j, k] = hidden[i, k] (pre-synaptic)
            hidden_inputs = hidden_outputs.transpose(1, 2)

            # 获取预计算的坐标
            (x_out, y_out), (x_in, y_in) = self.batched_hidden_coords

            # 步骤4：使用CPPN计算权重更新量并应用
            # delta_w = CPPN(坐标, pre激活, post激活, 当前权重)
            # W_hh(t+1) = W_hh(t) + delta_w
            self.hidden_to_hidden += self.delta_w_node(
                x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in,
                pre=hidden_inputs, post=hidden_outputs,
                w=self.hidden_to_hidden)
            
            # 可选：保存CPPN内部状态（未使用）
            # self.cppn_state = self.stateful_node.get_activs()

        # 移除额外的维度并返回
        # 形状: (batch_size, n_outputs, 1) -> (batch_size, n_outputs)
        return outputs.squeeze(2)

    @staticmethod
    def create(genome,
               config,

               input_coords,
               hidden_coords,
               output_coords,

               weight_threshold=0.2,
               activation=tanh_activation,
               batch_size=1,
               device='cuda:0'):
        """
        从NEAT基因组创建自适应递归网络（工厂方法）
        
        这是创建AdaptiveNet的标准方法。它将NEAT进化的基因组转换为
        可执行的PyTorch神经网络。
        
        工作流程：
        1. 使用create_cppn从基因组创建6个CPPN节点
        2. 这6个CPPN节点分别负责生成不同的权重矩阵：
           - w_ih: 输入->隐藏权重
           - b_h: 隐藏层偏置
           - w_hh: 隐藏->隐藏初始权重
           - b_o: 输出层偏置
           - w_ho: 隐藏->输出权重
           - delta_w: 隐藏->隐藏权重更新规则（自适应核心）
        3. 使用这些CPPN节点和坐标信息创建AdaptiveNet实例
        
        CPPN的输入：
        - x_in, y_in: 输入神经元的坐标
        - x_out, y_out: 输出神经元的坐标
        - pre: 输入神经元的激活值
        - post: 输出神经元的激活值
        - w: 当前权重值
        
        Args:
            genome: NEAT基因组，由NEAT算法进化得到
            config: NEAT配置对象
            input_coords (list): 输入神经元的2D坐标，如[[-1, 0], [0, 0], [1, 0]]
                                使用空间编码可以让CPPN利用几何规律性
            hidden_coords (list): 隐藏神经元的2D坐标
            output_coords (list): 输出神经元的2D坐标
            weight_threshold (float): 权重阈值，控制网络稀疏性
            activation (callable): 激活函数，默认tanh
            batch_size (int): 批量大小
            device (str): 计算设备
            
        Returns:
            AdaptiveNet: 创建的自适应递归网络实例
            
        使用示例：
            # 定义神经元坐标（2D空间排列）
            input_coords = [[-1, 0], [0, 0], [1, 0]]  # 3个输入
            hidden_coords = [[0, 1]]  # 1个隐藏神经元
            output_coords = [[0, 2]]  # 1个输出
            
            # 从基因组创建网络
            net = AdaptiveNet.create(
                genome, config,
                input_coords=input_coords,
                hidden_coords=hidden_coords,
                output_coords=output_coords,
                batch_size=4
            )
            
            # 使用网络
            outputs = net.activate(inputs)
        """
        # 从基因组创建6个CPPN节点
        # 每个节点是一个小型神经网络，接收7个输入，产生1个输出
        nodes = create_cppn(
            genome, config,
            ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],  # CPPN输入
            ['w_ih', 'b_h', 'w_hh', 'b_o', 'w_ho', 'delta_w'])  # CPPN输出

        # 解包6个CPPN节点
        w_ih_node = nodes[0]  # 生成输入->隐藏权重
        b_h_node = nodes[1]  # 生成隐藏层偏置
        w_hh_node = nodes[2]  # 生成隐藏->隐藏初始权重
        b_o_node = nodes[3]  # 生成输出层偏置
        w_ho_node = nodes[4]  # 生成隐藏->输出权重
        delta_w_node = nodes[5]  # 生成权重更新规则

        # 创建并返回AdaptiveNet实例
        return AdaptiveNet(w_ih_node,
                           b_h_node,
                           w_hh_node,
                           b_o_node,
                           w_ho_node,
                           delta_w_node,

                           input_coords,
                           hidden_coords,
                           output_coords,

                           weight_threshold=weight_threshold,
                           activation=activation,
                           batch_size=batch_size,
                           device=device)
