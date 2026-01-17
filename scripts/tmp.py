import os
import sys
import random
import argparse
from typing import List, Optional
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加当前目录到Python路径，以便导入你的训练模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_distributed(rank: int, world_size: int, master_addr: str, master_port: str):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)

    # 初始化进程组
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print(f"进程 {rank}/{world_size - 1} 初始化完成")


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def train_process(rank: int, world_size: int, args: argparse.Namespace, master_addr: str, master_port: str):
    """每个进程的训练函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size, master_addr, master_port)

    # 在这里导入你的训练代码
    # 注意：必须在setup_distributed之后导入，因为有些模块可能需要环境变量
    from benchmark_finetune import main as train_main

    # 执行训练
    train_main()

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分布式训练启动器")

    # 分布式配置
    parser.add_argument("--master_addr", type=str, default="127.0.0.1",
                        help="主节点地址")
    parser.add_argument("--master_port", type=str,
                        default=str(random.randint(20001, 29999)),
                        help="主节点端口")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="节点数量")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="每个节点的进程数量")

    # 训练参数
    parser.add_argument("--dataset_path", type=str,
                        default="/home/wsj/Desktop/code/github/Isaac-GR00T/demo_data/1128",
                        help="数据集路径")
    parser.add_argument("--modality_type", type=str, default="ymbot_d",
                        help="模态类型")
    parser.add_argument("--vlm_processor_path", type=str,
                        default="/home/wsj/Downloads/weights/qwen3-vl-2b",
                        help="VLM处理器路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="每个设备的训练批量大小")
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="全局批量大小")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="GPU数量")
    parser.add_argument("--deepspeed_config", type=str,
                        default="/home/wsj/Desktop/code/VLA/robot/config/deepspeed/zero3.json",
                        help="DeepSpeed配置路径")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置默认值
    if args.model_path is None:
        args.model_path = args.vlm_processor_path

    if args.num_gpus is None:
        args.num_gpus = args.nproc_per_node

    if args.global_batch_size is None:
        args.global_batch_size = args.nproc_per_node * args.per_device_train_batch_size

    # 打印配置
    print("=" * 50)
    print("分布式训练配置:")
    print(f"  主节点地址: {args.master_addr}")
    print(f"  主节点端口: {args.master_port}")
    print(f"  节点数量: {args.nnodes}")
    print(f"  每节点进程数: {args.nproc_per_node}")
    print(f"  GPU数量: {args.num_gpus}")
    print("\n训练参数:")
    for key, value in vars(args).items():
        if not key.startswith('master_') and key not in ['nnodes', 'nproc_per_node']:
            print(f"  {key}: {value}")
    print("=" * 50)

    # 设置环境变量（用于DeepSpeed等）
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(args.nnodes * args.nproc_per_node)

    mp.spawn(
        train_process,
        args=(args.nproc_per_node, args, args.master_addr, args.master_port),
        nprocs=args.nproc_per_node,
        join=True
    )

if __name__ == "__main__":
    main()