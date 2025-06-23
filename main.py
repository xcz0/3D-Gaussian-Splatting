#!/usr/bin/env python3
"""
3D Object Reconstruction using Gaussian Splatting
统一训练接口，通过配置文件管理不同的训练策略

主入口文件 - 简化版本
"""

import sys
import argparse
from pathlib import Path

from trainer import GaussianSplattingTrainer
from utils import list_available_configs


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting 训练启动器",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="示例:\n"
        f"  python {Path(__file__).name} --list\n"
        f"  python {Path(__file__).name} -c config/quick.yaml",
    )
    parser.add_argument("-c", "--config", type=str, help="配置文件路径")
    parser.add_argument(
        "-l", "--list", action="store_true", help="列出所有可用的配置文件"
    )
    args = parser.parse_args()

    if args.list:
        list_available_configs()
    elif args.config:
        try:
            trainer = GaussianSplattingTrainer(args.config)
            trainer.train()
        except KeyboardInterrupt:
            print("\n操作被用户中断。")
            sys.exit(130)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
