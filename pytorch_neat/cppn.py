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
CPPN（组合模式生成网络）模块

CPPN (Compositional Pattern Producing Networks) 是一种特殊的神经网络，
用于生成权重矩阵或权重更新规则。它是HyperNEAT算法的核心组件。

核心概念：
- CPPN不直接处理环境输入，而是根据神经元的空间位置生成权重
- 通过进化CPPN的拓扑和权重，间接进化出最终执行任务的神经网络
- CPPN可以利用几何对称性和规律性，生成具有结构的权重矩阵

主要用途：
1. HyperNEAT：生成固定的权重矩阵
2. 自适应网络：生成权重更新规则（Δw = CPPN(坐标, 激活值, 权重)）
3. 进化间接编码：减少搜索空间，利用几何规律性
"""

import torch
from neat.graphs import required_for_output

from .activations import str_to_activation
from .aggregations import str_to_aggregation


class Node:
    """
    CPPN中的计算节点类
    
    表示CPPN网络中的一个神经元。每个节点可以有多个输入（children），
    并对输入进行加权、聚合、然后应用激活函数。
    
    CPPN节点与普通神经网络节点的区别：
    - CPPN节点组成的网络用于生成权重，而不是直接处理任务输入
    - 节点的输入来自叶节点（Leaf），叶节点代表坐标、激活值等信息
    - 通过递归计算，最终产生权重值或权重更新值
    
    节点计算公式：
        output = activation(response * aggregation([w_i * child_i]) + bias)
    """
    def __init__(
        self,
        children,
        weights,
        response,
        bias,
        activation,
        aggregation,
        name=None,
        leaves=None,
    ):
        """
        初始化CPPN节点
        
        Args:
            children (list): 子节点列表（输入来源），可以是Node或Leaf
            weights (list): 对应每个子节点的连接权重
            response (float): 响应系数（缩放因子），乘以聚合后的值
            bias (float): 偏置项，加在激活函数之前
            activation (callable): 激活函数，如tanh、sigmoid等
            aggregation (callable): 聚合函数，如sum、product等
            name (str, optional): 节点名称，用于调试和标识
            leaves (dict, optional): 叶节点字典 {名称: Leaf对象}
                                     只有根节点需要保存此引用
        """
        self.children = children  # 输入节点列表
        self.leaves = leaves  # 叶节点字典（仅根节点使用）
        self.weights = weights  # 连接权重列表
        self.response = response  # 响应系数
        self.bias = bias  # 偏置项
        self.activation = activation  # 激活函数
        self.activation_name = activation  # 激活函数名称（用于打印）
        self.aggregation = aggregation  # 聚合函数
        self.aggregation_name = aggregation  # 聚合函数名称（用于打印）
        self.name = name  # 节点名称
        
        # 验证叶节点字典类型
        if leaves is not None:
            assert isinstance(leaves, dict)
        self.leaves = leaves
        
        # 缓存的激活值（避免重复计算）
        self.activs = None
        
        # 重置状态标志（用于递归重置）
        self.is_reset = None

    def __repr__(self):
        """
        返回节点的字符串表示（用于调试）
        
        格式：显示节点信息及其所有输入连接的树状结构
        
        Returns:
            str: 节点的详细字符串表示
        """
        # 节点头部信息
        header = "Node({}, response={}, bias={}, activation={}, aggregation={})".format(
            self.name,
            self.response,
            self.bias,
            self.activation_name,
            self.aggregation_name,
        )
        # 子节点信息（递归显示，带缩进）
        child_reprs = []
        for w, child in zip(self.weights, self.children):
            child_reprs.append(
                "    <- {} * ".format(w) + repr(child).replace("\n", "\n    ")
            )
        return header + "\n" + "\n".join(child_reprs)

    def activate(self, xs, shape):
        """
        计算节点的激活值
        
        执行标准的神经网络前向传播：加权、聚合、激活。
        
        计算流程：
            1. 加权：将每个输入乘以对应的权重 → [w_0*x_0, w_1*x_1, ...]
            2. 聚合：使用聚合函数合并所有加权输入 → sum([w_i*x_i])
            3. 缩放和偏置：response * 聚合值 + bias
            4. 激活：应用激活函数 → activation(response * 聚合值 + bias)
        
        Args:
            xs (list): 子节点的激活值列表，每个元素是torch.Tensor
            shape (tuple): 期望的输出形状（用于验证）
            
        Returns:
            torch.Tensor: 节点的激活值，形状为shape
            
        特殊情况：
            - 如果没有输入（xs为空），返回全为bias的张量
        """
        # 特殊情况：无输入节点，直接返回偏置
        if not xs:
            return torch.full(shape, self.bias)
        
        # 步骤1：对每个输入应用连接权重
        inputs = [w * x for w, x in zip(self.weights, xs)]
        
        try:
            # 步骤2：聚合所有加权输入（如求和）
            pre_activs = self.aggregation(inputs)
            
            # 步骤3+4：应用响应系数、偏置和激活函数
            activs = self.activation(self.response * pre_activs + self.bias)
            
            # 验证输出形状正确
            assert activs.shape == shape, "Wrong shape for node {}".format(self.name)
        except Exception:
            # 捕获并重新抛出异常，添加节点信息以便调试
            raise Exception("Failed to activate node {}".format(self.name))
        
        return activs

    def get_activs(self, shape):
        """
        获取节点的激活值（带缓存）
        
        使用记忆化（memoization）避免重复计算。如果已经计算过，
        直接返回缓存的结果；否则递归计算所有子节点，然后计算自身。
        
        Args:
            shape (tuple): 期望的输出形状
            
        Returns:
            torch.Tensor: 节点的激活值
        """
        # 如果还未计算过，递归计算
        if self.activs is None:
            # 获取所有子节点的激活值
            xs = [child.get_activs(shape) for child in self.children]
            # 计算自身激活值并缓存
            self.activs = self.activate(xs, shape)
        return self.activs

    def __call__(self, **inputs):
        """
        调用CPPN节点，根据输入生成输出
        
        这是CPPN的主要接口。给定叶节点的值（如坐标、激活值等），
        计算CPPN的输出（如权重值或权重更新值）。
        
        使用方法：
            cppn_node(x_in=..., y_in=..., x_out=..., y_out=..., pre=..., post=..., w=...)
        
        Args:
            **inputs: 叶节点的输入值，键名对应叶节点名称
                     例如：x_in, y_in, x_out, y_out（神经元坐标）
                          pre, post（神经元激活值）
                          w（当前权重值）
                          
        Returns:
            torch.Tensor: CPPN的输出值（权重或权重更新）
            
        流程：
            1. 重置所有节点的缓存
            2. 将输入值设置到对应的叶节点
            3. 递归计算输出节点的激活值
        """
        # 确保这是根节点（有叶节点引用）
        assert self.leaves is not None
        # 确保提供了输入
        assert inputs
        
        # 从第一个输入推断形状（所有输入应该有相同的形状）
        shape = list(inputs.values())[0].shape
        
        # 重置网络（清除缓存）
        self.reset()
        
        # 将输入值设置到对应的叶节点
        for name in self.leaves.keys():
            # 验证输入形状一致
            assert (
                inputs[name].shape == shape
            ), "Wrong activs shape for leaf {}, {} != {}".format(
                name, inputs[name].shape, shape
            )
            self.leaves[name].set_activs(inputs[name])
        
        # 递归计算并返回输出
        return self.get_activs(shape)

    def _prereset(self):
        """
        重置准备阶段：初始化重置标志
        
        三阶段重置的第一阶段，递归地将所有节点的is_reset标志设为False。
        这样可以正确处理有向无环图（DAG）中的共享节点。
        """
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:
                child._prereset()  # pylint: disable=protected-access

    def _postreset(self):
        """
        重置清理阶段：清除重置标志
        
        三阶段重置的第三阶段，将所有节点的is_reset标志设回None。
        恢复初始状态，为下次重置做准备。
        """
        if self.is_reset is not None:
            self.is_reset = None
            for child in self.children:
                child._postreset()  # pylint: disable=protected-access

    def _reset(self):
        """
        重置执行阶段：清除缓存的激活值
        
        三阶段重置的第二阶段，递归地清除所有节点的激活值缓存。
        使用is_reset标志避免重复重置（在DAG中某些节点可能有多个父节点）。
        """
        if not self.is_reset:
            self.is_reset = True
            self.activs = None  # 清除缓存
            for child in self.children:
                child._reset()  # pylint: disable=protected-access

    def reset(self):
        """
        重置节点及其所有子节点
        
        清除所有缓存的激活值，使得下次调用时重新计算。
        
        为什么使用三阶段重置？
        - CPPN可能是有向无环图（DAG），节点可能被多次访问
        - 三阶段重置确保每个节点恰好被重置一次
        - 避免无限递归和重复工作
        """
        self._prereset()  # 阶段1：初始化标志
        self._reset()  # 阶段2：清除缓存
        self._postreset()  # 阶段3：清理标志


class Leaf:
    """
    CPPN的叶节点类
    
    叶节点是CPPN的输入端，不进行任何计算，只是存储和传递输入值。
    
    在HyperNEAT和自适应网络中，叶节点通常表示：
    - x_in, y_in: 输入神经元的2D空间坐标
    - x_out, y_out: 输出神经元的2D空间坐标
    - pre: 输入神经元的激活值
    - post: 输出神经元的激活值
    - w: 当前的连接权重
    
    CPPN根据这些信息生成权重值或权重更新值。
    """
    def __init__(self, name=None):
        """
        初始化叶节点
        
        Args:
            name (str, optional): 叶节点的名称，如'x_in', 'y_out'等
        """
        self.activs = None  # 存储的输入值（张量）
        self.name = name  # 节点名称

    def __repr__(self):
        """返回叶节点的字符串表示"""
        return "Leaf({})".format(self.name)

    def set_activs(self, activs):
        """
        设置叶节点的值
        
        Args:
            activs (torch.Tensor): 要设置的值
        """
        self.activs = activs

    def get_activs(self, shape):
        """
        获取叶节点的值（带验证）
        
        Args:
            shape (tuple): 期望的形状
            
        Returns:
            torch.Tensor: 叶节点存储的值
            
        Raises:
            AssertionError: 如果值未设置或形状不匹配
        """
        # 确保值已设置
        assert self.activs is not None, "Missing activs for leaf {}".format(self.name)
        # 确保形状正确
        assert (
            self.activs.shape == shape
        ), "Wrong activs shape for leaf {}, {} != {}".format(
            self.name, self.activs.shape, shape
        )
        return self.activs

    def _prereset(self):
        """叶节点的重置准备（空操作）"""
        pass

    def _postreset(self):
        """叶节点的重置清理（空操作）"""
        pass

    def _reset(self):
        """叶节点的重置执行：清除存储的值"""
        self.activs = None

    def reset(self):
        """重置叶节点，清除存储的值"""
        self._reset()


def create_cppn(genome, config, leaf_names, node_names, output_activation=None):
    """
    从NEAT基因组创建CPPN网络
    
    这是将NEAT进化的基因组转换为可执行CPPN的核心函数。
    NEAT负责进化网络拓扑和权重，本函数负责构建实际的计算图。
    
    Args:
        genome: NEAT基因组，包含节点和连接信息
        config: NEAT配置对象，包含输入输出节点的定义
        leaf_names (list): 叶节点名称列表，如['x_in', 'y_in', 'x_out', 'y_out', ...]
                          长度必须等于genome中输入节点的数量
        node_names (list): 输出节点名称列表，如['w_ih', 'b_h', 'delta_w', ...]
                          长度必须等于genome中输出节点的数量
        output_activation (callable, optional): 覆盖输出节点的激活函数
                                               如果为None，使用基因组中定义的激活函数
                                               
    Returns:
        list: CPPN输出节点列表，每个节点对应一个输出功能
              可以通过调用 node(x_in=..., y_in=..., ...) 来计算输出
              
    工作流程：
        1. 从基因组中提取连接信息
        2. 为每个输入创建Leaf节点
        3. 递归构建中间层和输出层的Node节点
        4. 返回输出节点列表
        
    示例：
        # 创建生成权重的CPPN
        nodes = create_cppn(genome, config, 
                           ['x_in', 'y_in', 'x_out', 'y_out'],
                           ['weight'])
        weight_node = nodes[0]
        
        # 使用CPPN生成权重矩阵
        weights = weight_node(x_in=x_coords_in, y_in=y_coords_in,
                             x_out=x_coords_out, y_out=y_coords_out)
    """
    genome_config = config.genome_config
    
    # 找出对生成输出必需的所有节点（剪枝不需要的节点）
    required = required_for_output(
        genome_config.input_keys, genome_config.output_keys, genome.connections
    )

    # 收集所有激活的连接，构建节点的输入关系
    # node_inputs[node_id] = [(input_id, weight), ...]
    node_inputs = {i: [] for i in genome_config.output_keys}
    for cg in genome.connections.values():
        # 跳过未激活的连接
        if not cg.enabled:
            continue

        i, o = cg.key  # 连接：从节点i到节点o
        
        # 跳过不必需的连接（不影响输出）
        if o not in required and i not in required:
            continue

        # 跳过从输出节点发出的连接（输出节点不应该作为输入）
        if i in genome_config.output_keys:
            continue

        # 记录连接信息
        if o not in node_inputs:
            node_inputs[o] = [(i, cg.weight)]
        else:
            node_inputs[o].append((i, cg.weight))

        # 确保输入节点在字典中（即使它没有输入）
        if i not in node_inputs:
            node_inputs[i] = []

    # 为所有输入节点创建Leaf对象
    nodes = {i: Leaf() for i in genome_config.input_keys}

    # 验证叶节点名称数量正确
    assert len(leaf_names) == len(genome_config.input_keys)
    
    # 创建叶节点字典：名称 -> Leaf对象
    leaves = {name: nodes[i] for name, i in zip(leaf_names, genome_config.input_keys)}

    def build_node(idx):
        """
        递归构建节点
        
        使用深度优先搜索，从输出节点开始递归构建整个CPPN。
        已构建的节点会被缓存，避免重复构建（DAG共享节点）。
        
        Args:
            idx: 节点ID
            
        Returns:
            Node或Leaf: 构建的节点对象
        """
        # 如果已经构建过，直接返回（叶节点或已构建的中间节点）
        if idx in nodes:
            return nodes[idx]
        
        # 从基因组获取节点基因
        node = genome.nodes[idx]
        
        # 获取该节点的所有输入连接
        conns = node_inputs[idx]
        
        # 递归构建所有子节点
        children = [build_node(i) for i, w in conns]
        
        # 提取连接权重
        weights = [w for i, w in conns]
        
        # 确定激活函数（输出节点可以被覆盖）
        if idx in genome_config.output_keys and output_activation is not None:
            activation = output_activation
        else:
            activation = str_to_activation[node.activation]
        
        # 确定聚合函数
        aggregation = str_to_aggregation[node.aggregation]
        
        # 创建Node对象
        nodes[idx] = Node(
            children,
            weights,
            node.response,
            node.bias,
            activation,
            aggregation,
            leaves=leaves,  # 所有节点共享叶节点字典
        )
        return nodes[idx]

    # 构建所有输出节点（会递归构建所有必需的中间节点）
    for idx in genome_config.output_keys:
        build_node(idx)

    # 收集输出节点
    outputs = [nodes[i] for i in genome_config.output_keys]

    # 设置叶节点的名称（用于调试）
    for name in leaf_names:
        leaves[name].name = name

    # 设置输出节点的名称（用于调试）
    for i, name in zip(genome_config.output_keys, node_names):
        nodes[i].name = name

    return outputs


def clamp_weights_(weights, weight_threshold=0.2, weight_max=3.0):
    """
    对权重矩阵进行阈值处理和限幅（原地操作）
    
    这个函数执行两个重要的权重处理操作：
    1. 阈值化：将绝对值小于阈值的权重设为0（稀疏化）
    2. 限幅：将权重限制在[-weight_max, weight_max]范围内
    
    权重处理流程：
        1. 找出绝对值 < weight_threshold 的权重
        2. 将这些权重设为0（剪枝不重要的连接）
        3. 正权重减去阈值（保持稀疏性）
        4. 负权重加上阈值（保持稀疏性）
        5. 限制在[-weight_max, weight_max]范围内
    
    为什么这样做？
    - 稀疏化：减少连接数量，提高计算效率，减少过拟合
    - 限幅：防止权重爆炸，保持数值稳定性
    - 阈值平移：确保非零权重有最小强度
    
    Args:
        weights (torch.Tensor): 权重矩阵，会被原地修改
        weight_threshold (float): 权重阈值，绝对值小于此值的权重被设为0
        weight_max (float): 权重最大绝对值
        
    示例：
        假设 weight_threshold=0.2, weight_max=3.0
        - 0.1 → 0 (小于阈值，被剪枝)
        - 0.5 → 0.3 (减去阈值)
        - -0.5 → -0.3 (加上阈值)
        - 4.0 → 3.0 (限幅)
        - -4.0 → -3.0 (限幅)
    """
    # TODO: also try LEO (另一种权重处理方法)
    
    # 步骤1：找出绝对值小于阈值的权重
    low_idxs = weights.abs() < weight_threshold
    
    # 步骤2：将小权重设为0（剪枝）
    weights[low_idxs] = 0
    
    # 步骤3：正权重减去阈值（平移）
    weights[weights > 0] -= weight_threshold
    
    # 步骤4：负权重加上阈值（平移）
    weights[weights < 0] += weight_threshold
    
    # 步骤5：限制在最大值范围内
    weights[weights > weight_max] = weight_max
    weights[weights < -weight_max] = -weight_max


def get_coord_inputs(in_coords, out_coords, batch_size=None):
    """
    生成CPPN的坐标输入
    
    将输入和输出神经元的坐标扩展成适合CPPN处理的形状。
    CPPN需要为每对(输入神经元, 输出神经元)生成一个权重值，
    因此需要将坐标广播成矩阵形式。
    
    输入形状：
        - in_coords: (n_in, 2) 或 (batch_size, n_in, 2)
        - out_coords: (n_out, 2) 或 (batch_size, n_out, 2)
        
    输出形状（无batch）：
        - x_out, y_out, x_in, y_in: (n_out, n_in)
        
    输出形状（有batch）：
        - x_out, y_out, x_in, y_in: (batch_size, n_out, n_in)
    
    这样，位置[i, j]对应从输入神经元j到输出神经元i的连接坐标。
    
    Args:
        in_coords (torch.Tensor): 输入神经元坐标，形状为(n_in, 2)
        out_coords (torch.Tensor): 输出神经元坐标，形状为(n_out, 2)
        batch_size (int, optional): 批量大小，如果提供则生成批量坐标
        
    Returns:
        tuple: ((x_out, y_out), (x_in, y_in))
               四个张量，每个形状为(n_out, n_in)或(batch_size, n_out, n_in)
               
    示例：
        假设有2个输入神经元[(0, 0), (1, 0)]和3个输出神经元[(0, 1), (1, 1), (2, 1)]
        
        x_out = [[0, 0],      y_out = [[1, 1],
                 [1, 1],               [1, 1],
                 [2, 2]]               [1, 1]]
                 
        x_in =  [[0, 1],      y_in =  [[0, 0],
                 [0, 1],               [0, 0],
                 [0, 1]]               [0, 0]]
                 
        这样x_out[i,j], y_out[i,j], x_in[i,j], y_in[i,j]
        就表示从输入j到输出i的连接的坐标信息。
    """
    n_in = len(in_coords)
    n_out = len(out_coords)

    if batch_size is not None:
        # 批量模式：扩展坐标以支持批量处理
        in_coords = in_coords.unsqueeze(0).expand(batch_size, n_in, 2)
        out_coords = out_coords.unsqueeze(0).expand(batch_size, n_out, 2)

        # 提取并广播x_out（输出神经元的x坐标）
        x_out = out_coords[:, :, 0].unsqueeze(2).expand(batch_size, n_out, n_in)
        # 提取并广播y_out（输出神经元的y坐标）
        y_out = out_coords[:, :, 1].unsqueeze(2).expand(batch_size, n_out, n_in)
        # 提取并广播x_in（输入神经元的x坐标）
        x_in = in_coords[:, :, 0].unsqueeze(1).expand(batch_size, n_out, n_in)
        # 提取并广播y_in（输入神经元的y坐标）
        y_in = in_coords[:, :, 1].unsqueeze(1).expand(batch_size, n_out, n_in)
    else:
        # 非批量模式
        # 提取并广播x_out
        x_out = out_coords[:, 0].unsqueeze(1).expand(n_out, n_in)
        # 提取并广播y_out
        y_out = out_coords[:, 1].unsqueeze(1).expand(n_out, n_in)
        # 提取并广播x_in
        x_in = in_coords[:, 0].unsqueeze(0).expand(n_out, n_in)
        # 提取并广播y_in
        y_in = in_coords[:, 1].unsqueeze(0).expand(n_out, n_in)

    return (x_out, y_out), (x_in, y_in)
