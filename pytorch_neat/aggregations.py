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
聚合函数模块：定义如何聚合多个输入信号

在神经网络中，当一个神经元接收多个输入时，需要将这些输入聚合成单个值。
本模块提供了常用的聚合函数。

在NEAT中，每个节点可以指定其聚合函数，影响信息如何在网络中流动。
"""

from functools import reduce  # 用于序列的累积操作
from operator import mul  # 乘法运算符
import torch


def sum_aggregation(inputs):
    """
    求和聚合函数（最常用）
    
    将所有输入信号相加。这是神经网络中最标准的聚合方式。
    
    公式: output = x₁ + x₂ + ... + xₙ
    
    Args:
        inputs: 可迭代对象，包含要聚合的值（可以是标量或torch.Tensor）
        
    Returns:
        float/int 或 torch.Tensor: 所有输入的总和
        
    示例:
        >>> sum_aggregation([1, 2, 3])
        6
        >>> sum_aggregation([-1, 1, 0])
        0
    """
    if not inputs:
        return 0
    # 检查第一个元素是否是torch.Tensor
    if isinstance(inputs[0], torch.Tensor):
        # 对于张量，确保所有张量都在同一设备上
        # 获取第一个张量的设备
        device = inputs[0].device
        # 将所有张量移动到同一设备并求和
        result = inputs[0]
        for x in inputs[1:]:
            if isinstance(x, torch.Tensor):
                result = result + x.to(device)
            else:
                result = result + x
        return result
    else:
        # 对于标量，使用Python内置的sum函数
        return sum(inputs)


def prod_aggregation(inputs):
    """
    乘积聚合函数
    
    将所有输入信号相乘。这种聚合方式可以实现"与"逻辑（任一输入为0则输出为0）。
    
    公式: output = x₁ × x₂ × ... × xₙ
    
    Args:
        inputs: 可迭代对象，包含要聚合的值（可以是标量或torch.Tensor）
        
    Returns:
        float/int 或 torch.Tensor: 所有输入的乘积
        
    注意:
        - 如果inputs为空，返回1（乘法的单位元）
        - 任何输入为0，结果为0
        - 可能导致数值溢出或下溢
        
    示例:
        >>> prod_aggregation([2, 3, 4])
        24
        >>> prod_aggregation([1, 0, 5])
        0
        >>> prod_aggregation([])
        1
    """
    if not inputs:
        return 1
    # 检查第一个元素是否是torch.Tensor
    if isinstance(inputs[0], torch.Tensor):
        # 对于张量，确保所有张量都在同一设备上
        # 获取第一个张量的设备
        device = inputs[0].device
        # 将所有张量移动到同一设备并相乘
        result = inputs[0]
        for x in inputs[1:]:
            if isinstance(x, torch.Tensor):
                result = result * x.to(device)
            else:
                result = result * x
        return result
    else:
        # 使用reduce函数累积相乘所有元素
        # reduce(mul, inputs, 1) 等价于：
        # result = 1
        # for x in inputs:
        #     result = result * x
        # return result
        return reduce(mul, inputs, 1)


# 字符串到聚合函数的映射字典
# 用于从NEAT配置文件中根据名称选择聚合函数
str_to_aggregation = {
    'sum': sum_aggregation,   # 求和（默认）
    'prod': prod_aggregation, # 乘积
}
