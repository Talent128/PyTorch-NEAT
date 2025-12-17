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


def generate_results_dir_name(scenario_name: str, algorithm_name: str, task_config, seed: int) -> str:
    """根据任务配置参数生成结果目录名
    
    格式: results/task_name_algorithm_name_param1_param2_..._seed{seed}
    例如: results/flocking_recurrent_200_4_3_-0.1_True_seed42
    
    Args:
        scenario_name (str): 场景/任务名称，如 'flocking'
        algorithm_name (str): 算法名称，如 'recurrent'
        task_config: 任务配置对象（dataclass 或普通对象）
        seed (int): 随机种子
        
    Returns:
        str: 结果目录路径
    """
    from dataclasses import fields, is_dataclass
    
    # 获取任务配置参数值（按字段顺序）
    task_params = []
    if is_dataclass(task_config):
        for field in fields(task_config):
            value = getattr(task_config, field.name)
            task_params.append(str(value))
    else:
        # 如果不是dataclass，尝试按排序键顺序
        for key in sorted(vars(task_config).keys()):
            value = getattr(task_config, key)
            task_params.append(str(value))
    
    # 构建目录名: task_algorithm_params_seed
    params_str = "_".join(task_params)
    dir_name = f"{scenario_name}_{algorithm_name}_{params_str}_seed{seed}"
    
    return f"results/{dir_name}"


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

