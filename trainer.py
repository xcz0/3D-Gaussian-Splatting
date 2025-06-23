#!/usr/bin/env python3
"""
3D Gaussian Splatting 训练器核心模块
"""

import sys
import time
from pathlib import Path
from typing import Optional

from logger import Logger
from utils import (
    load_config,
    validate_paths,
    chdir_context,
    setup_environment,
    get_resume_iteration,
    print_final_results,
)
from commands import CommandBuilder, CommandExecutor


class GaussianSplattingTrainer:
    """3D Gaussian Splatting 训练器"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = Logger(self.__class__.__name__)
        self.config = load_config(self.config_path)
        self.gaussian_splatting_dir = Path(__file__).parent / "gaussian-splatting"
        self.start_time = None

        # 初始化命令构建器和执行器
        self.command_builder = CommandBuilder(self.config)
        self.command_executor = CommandExecutor(self.config)

    def train(self):
        """执行完整的训练、渲染和评估流程"""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"开始训练流程: {self.config.name}")
        self.logger.info(f"描述: {self.config.description}")
        self.logger.info("=" * 60)

        try:
            validate_paths(self.config, self.gaussian_splatting_dir)
            with chdir_context(self.gaussian_splatting_dir):
                setup_environment(self.config, self.gaussian_splatting_dir)
                self._execute_training_flow()
            print_final_results(self.config, self.start_time)

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            self.logger.error(f"流程中断: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"发生未知错误: {e}")
            sys.exit(1)

    def _execute_training_flow(self):
        """执行分段训练和中间评估的核心逻辑"""
        # 检查是否需要继续训练
        resume_iter = get_resume_iteration(self.config)
        total_iters = self.config.training.iterations

        if resume_iter >= total_iters:
            self.logger.info(
                f"模型已完成训练 (当前: {resume_iter}, 目标: {total_iters})，跳过训练阶段"
            )
            self.logger.info("直接执行最终的渲染和评估...")
            self._run_rendering()
            self._run_metrics()
            return

        if resume_iter > 0:
            self.logger.info(
                f"将从第 {resume_iter} 次迭代继续训练到第 {total_iters} 次"
            )

        if not self.config.post_processing.enable_intermediate_eval:
            self.logger.info("中间评估已禁用，将执行单次完整训练。")
            self._run_training_segment(
                resume_iter, total_iters, is_first_segment=(resume_iter == 0)
            )
        else:
            # 计算所有需要评估的迭代点
            eval_points = sorted(
                set(
                    self.config.training.test_iterations
                    + self.config.training.save_iterations
                )
            )

            # 过滤超出范围的迭代点和已完成的迭代点
            eval_points = [p for p in eval_points if resume_iter < p < total_iters]
            if not eval_points or max(eval_points) < total_iters:
                eval_points.append(total_iters)

            self.logger.info(f"将在以下迭代点进行评估: {eval_points}")

            start_iter = resume_iter
            for i, end_iter in enumerate(eval_points):
                if start_iter >= end_iter:
                    continue

                self.logger.info("-" * 50)
                self.logger.info(f"开始训练段: {start_iter} -> {end_iter}")
                self._run_training_segment(
                    start_iter, end_iter, is_first_segment=(start_iter == 0)
                )

                self.logger.info(f"在第 {end_iter} 次迭代进行评估...")
                self._run_rendering(iteration=end_iter)
                self._run_metrics(iteration=end_iter)
                start_iter = end_iter

        # 如果训练完成后需要，执行最终的渲染和评估（通常用于覆盖默认的test/train目录）
        self.logger.info("-" * 50)
        self.logger.info("执行最终的渲染和评估...")
        self._run_rendering()
        self._run_metrics()

    def _run_training_segment(
        self, start_iter: int, end_iter: int, is_first_segment: bool
    ):
        """运行单段训练"""
        cmd = self.command_builder.build_train_command(
            start_iter, end_iter, is_first_segment
        )
        self.command_executor.run_training_segment(cmd, start_iter, end_iter)

    def _run_rendering(self, iteration: Optional[int] = None):
        """运行渲染脚本 (render.py)"""
        cmd = self.command_builder.build_render_command(iteration)
        self.command_executor.run_rendering(cmd, iteration)

    def _run_metrics(self, iteration: Optional[int] = None):
        """运行评估脚本 (metrics.py)"""
        cmd = self.command_builder.build_metrics_command(iteration)
        self.command_executor.run_metrics(cmd, iteration)
