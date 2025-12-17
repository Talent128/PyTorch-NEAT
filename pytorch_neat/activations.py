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
激活函数模块：提供神经网络常用的激活函数
注意：大部分激活函数都进行了缩放（乘以常数），以增强信号强度。
"""

import torch
import torch.nn.functional as F


def sigmoid_activation(x):
    """
    Sigmoid激活函数（放大版）
    
    公式: σ(5x) = 1 / (1 + e^(-5x))
    输出范围: (0, 1)
    
    特点：
    - 输出有界，适合作为概率输出
    - 乘以5使函数更陡峭，增强信号
    - 易饱和（梯度消失），但在NEAT中不影响（无反向传播）
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，值域(0, 1)
    """
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    """
    双曲正切激活函数（放大版）
    
    公式: tanh(2.5x)
    输出范围: (-1, 1)
    
    特点：
    - 输出以0为中心，比sigmoid更好
    - 乘以2.5使函数更陡峭
    - 适合大多数任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，值域(-1, 1)
    """
    return torch.tanh(2.5 * x)


def abs_activation(x):
    """
    绝对值激活函数
    
    公式: |x|
    输出范围: [0, ∞)
    
    特点：
    - 非单调函数，可以产生有趣的行为
    - 在x=0处不可微（但NEAT不需要梯度）
    - 适合某些特殊任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，所有值非负
    """
    return torch.abs(x)


def gauss_activation(x):
    """
    高斯（径向基）激活函数
    
    公式: exp(-5x²)
    输出范围: (0, 1]
    
    特点：
    - 在x=0处最大值为1
    - 对称的钟形曲线
    - 可以检测输入是否接近0
    - 适合模式识别任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，在x=0时最大
    """
    return torch.exp(-5.0 * x**2)


def identity_activation(x):
    """
    恒等激活函数（线性）
    
    公式: f(x) = x
    输出范围: (-∞, ∞)
    
    特点：
    - 不改变输入
    - 保持线性特性
    - 常用于输出层或CPPN
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 输入本身
    """
    return x


def sin_activation(x):
    """
    正弦激活函数
    
    公式: sin(x)
    输出范围: [-1, 1]
    
    特点：
    - 周期性函数，可以产生振荡行为
    - 在CPPN中很有用（生成重复模式）
    - 可以帮助探索更复杂的行为
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，周期性振荡
    """
    return torch.sin(x)


def relu_activation(x):
    """
    ReLU（修正线性单元）激活函数
    
    公式: max(0, x)
    输出范围: [0, ∞)
    
    特点：
    - 深度学习中最常用的激活函数
    - 计算简单，不会饱和（正值部分）
    - 可能导致"神经元死亡"（负值全为0）
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，负值变为0
    """
    return F.relu(x)


# 字符串到激活函数的映射字典
# 用于从配置文件中根据名称选择激活函数
str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
}
