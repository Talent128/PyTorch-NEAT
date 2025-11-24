"""
训练入口

使用Hydra进行配置管理，启动NEAT进化训练
"""
# 导入__init__.py以注册ConfigStore配置
import __init__  # noqa

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra_config import load_experiment_from_hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主函数
    
    使用方式:
        python train.py algorithm=recurrent task=vmas/transport
        python train.py algorithm=adaptive_linear task=vmas/navigation
        python train.py algorithm=adaptive task=vmas/transport seed=42
    
    Args:
        cfg (DictConfig): Hydra配置字典
    """
    # 获取运行时选择
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))
    print(f"{'='*60}\n")
    
    # 创建实验
    experiment = load_experiment_from_hydra(cfg, task_name, algorithm_name)
    
    # 将Hydra配置目录移动到results_dir
    import os
    import shutil
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    hydra_dir = os.path.join(hydra_output_dir, '.hydra')
    target_hydra_dir = os.path.join(experiment.config.results_dir, '.hydra')
    
    if os.path.exists(hydra_dir) and not os.path.exists(target_hydra_dir):
        shutil.copytree(hydra_dir, target_hydra_dir)
        print(f"Hydra配置已复制到: {target_hydra_dir}\n")
    
    # 运行训练
    experiment.run()


if __name__ == "__main__":
    main()
