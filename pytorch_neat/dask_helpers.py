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
Dask分布式计算辅助模块

Dask是一个灵活的Python并行计算库。本模块提供了设置Dask客户端的辅助函数，
用于在集群上分布式运行NEAT评估，显著加速大规模实验。

使用场景：
- 在多台机器上并行评估大量基因组
- 利用集群资源加速训练
- 处理计算密集型任务
"""

import time  # 用于重试之间的延迟

from dask.distributed import Client  # Dask分布式客户端


def setup_dask(scheduler, retries=-1):
    """
    设置并连接到Dask集群
    
    这个函数尝试连接到Dask调度器，支持本地集群和远程集群。
    如果连接失败，会自动重试直到成功或达到最大重试次数。
    
    Args:
        scheduler (str or None): Dask调度器地址
            - None 或 "{scheduler}": 创建本地集群
            - "tcp://host:port": 连接到远程调度器
        retries (int): 最大重试次数
            - -1: 无限重试（默认）
            - n > 0: 最多重试n次
            
    Returns:
        Client: Dask分布式客户端对象
        
    Raises:
        Exception: 达到最大重试次数后仍无法连接
        
    示例:
        >>> # 本地集群（适合单机多核）
        >>> client = setup_dask(None)
        
        >>> # 连接到远程集群
        >>> client = setup_dask("tcp://192.168.1.100:8786")
        
        >>> # 限制重试次数
        >>> client = setup_dask("tcp://cluster:8786", retries=5)
    """
    # 检查是否使用本地集群
    if scheduler is None or scheduler == "{scheduler}":
        print("Setting up local cluster...")  # 提示用户正在创建本地集群
        return Client()  # 创建本地Dask集群客户端
    
    # 连接到远程调度器
    succeeded = False  # 连接成功标志
    try_num = 0  # 当前尝试次数
    
    # 循环尝试连接，直到成功或达到最大重试次数
    while not succeeded:
        try_num += 1  # 递增尝试次数
        
        # 检查是否达到最大重试次数
        if try_num == retries:
            raise Exception("Failed to connect to Dask client")  # 抛出异常
        
        try:
            # 尝试连接到调度器，超时时间60秒
            client = Client(scheduler, timeout=60)
            succeeded = True  # 连接成功，设置标志
        except Exception as e:  # pylint: disable=broad-except
            # 连接失败，打印错误信息
            print(e)
        
        # 等待15秒后重试（避免过于频繁的重试）
        time.sleep(15)

    # 返回成功连接的客户端
    return client
