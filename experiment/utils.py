import random
import numpy as np
import torch
import tempfile
import os


def seed_everything(seed: int) -> None:
    """设置所有随机数生成器的种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_circular_coords(n, radius=1.0):
    """在圆周上均匀分布生成坐标
    
    Args:
        n (int): 坐标点数量
        radius (float): 圆的半径
        
    Returns:
        list: 坐标列表，每个元素为[x, y]
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return [[radius * np.cos(a), radius * np.sin(a)] for a in angles]


def generate_grid_coords(n):
    """在网格上生成坐标
    
    Args:
        n (int): 坐标点数量
        
    Returns:
        list: 坐标列表，每个元素为[x, y]
    """
    side = int(np.ceil(np.sqrt(n)))
    coords = []
    for i in range(n):
        x = (i % side) / max(side - 1, 1) * 2 - 1  # [-1, 1]
        y = (i // side) / max(side - 1, 1) * 2 - 1
        coords.append([x, y])
    return coords[:n]


def load_neat_config_with_substitution(cfg_path, num_inputs, num_outputs, output_path=None):
    """加载NEAT配置文件并替换占位符
    
    Args:
        cfg_path (str): NEAT配置文件路径(.cfg)
        num_inputs (int): 输入神经元数量
        num_outputs (int): 输出神经元数量
        output_path (str, optional): 输出文件路径，如果为None则创建临时文件
        
    Returns:
        str: 配置文件路径
    """
    # 读取配置文件
    with open(cfg_path, 'r') as f:
        content = f.read()
    
    # 替换占位符
    content = content.replace('{num_inputs}', str(num_inputs))
    content = content.replace('{num_outputs}', str(num_outputs))
    
    # 保存到指定路径或临时文件
    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path
    else:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(content)
            temp_path = f.name
        return temp_path

