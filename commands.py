#!/usr/bin/env python3
"""
命令构建和执行模块
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from config import AppConfig
from logger import Logger


class CommandBuilder:
    """命令构建器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = Logger(self.__class__.__name__)
        # 确保路径是绝对路径
        self._normalize_paths()

    def _normalize_paths(self):
        """将配置中的相对路径转换为绝对路径"""
        # 转换源路径为绝对路径
        source_path = Path(self.config.paths.source_path)
        if not source_path.is_absolute():
            self.config.paths.source_path = str(Path.cwd() / source_path)

        # 转换模型路径为绝对路径
        model_path = Path(self.config.paths.model_path)
        if not model_path.is_absolute():
            self.config.paths.model_path = str(Path.cwd() / model_path)

    def build_train_command(
        self, start_iter: int, end_iter: int, is_first_segment: bool
    ) -> List[str]:
        """构建 train.py 的命令行参数"""
        cfg = self.config.training
        cmd = ["uv", "run", "python", "train.py"]

        # 参数映射: (dataclass 属性, 命令行标志)
        arg_map = {
            "source_path": "--source_path",
            "model_path": "--model_path",
            "iterations": "--iterations",
            "densify_grad_threshold": "--densify_grad_threshold",
            "densification_interval": "--densification_interval",
            "densify_until_iter": "--densify_until_iter",
            "opacity_reset_interval": "--opacity_reset_interval",
            "position_lr_init": "--position_lr_init",
            "position_lr_final": "--position_lr_final",
            "scaling_lr": "--scaling_lr",
            "resolution": "--resolution",
            "lambda_dssim": "--lambda_dssim",
            "data_device": "--data_device",
            "optimizer_type": "--optimizer_type",
        }

        # 动态添加参数
        params = {
            **self.config.paths.__dict__,
            **cfg.__dict__,
            "iterations": end_iter,  # 覆盖迭代次数
        }

        for attr, flag in arg_map.items():
            if (value := params.get(attr)) is not None:
                cmd.extend([flag, str(value)])

        # 布尔标志
        for flag in ["eval", "quiet", "disable_viewer"]:
            if getattr(cfg, flag, False):
                cmd.append(f"--{flag}")

        # 列表参数
        for attr in ["test_iterations", "save_iterations", "checkpoint_iterations"]:
            if values := getattr(cfg, attr, []):
                cmd.extend([f"--{attr}"] + [str(v) for v in values])

        # 继续训练的逻辑
        if start_iter > 0 or not is_first_segment:
            # 构建正确的检查点文件路径
            checkpoint_file = f"{self.config.paths.model_path}/chkpnt{start_iter}.pth"
            # 检查检查点文件是否存在
            if Path(checkpoint_file).exists():
                cmd.extend(["--start_checkpoint", checkpoint_file])
                self.logger.info(f"将从检查点继续训练: {checkpoint_file}")
            else:
                self.logger.warning(
                    f"检查点文件不存在: {checkpoint_file}，将跳过继续训练"
                )
                # 如果检查点不存在但不是第一段，这可能是个问题
                if not is_first_segment:
                    self.logger.error("中间训练段缺少必要的检查点文件")
            # 注意：原始Gaussian Splatting不支持--start_iteration参数
            # 训练会从checkpoint自动继续

        return cmd

    def build_render_command(self, iteration: Optional[int] = None) -> List[str]:
        """构建渲染命令"""
        cmd = [
            "uv",
            "run",
            "python",
            "render.py",
            "--model_path",
            self.config.paths.model_path,
            "--source_path",
            self.config.paths.source_path,
        ]
        if iteration:
            cmd.extend(["--iteration", str(iteration)])
        if self.config.training.quiet:
            cmd.append("--quiet")
        return cmd

    def build_metrics_command(self, iteration: Optional[int] = None) -> List[str]:
        """构建评估命令"""
        cmd = [
            "uv",
            "run",
            "python",
            "metrics.py",
            "--model_paths",
            self.config.paths.model_path,
        ]
        # 注意：原始的metrics.py不支持--iteration参数
        # 它会自动评估model_path/test目录下的所有渲染结果
        return cmd


class CommandExecutor:
    """命令执行器"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = Logger(self.__class__.__name__)

    def run_training_segment(self, cmd: List[str], start_iter: int, end_iter: int):
        """运行单段训练"""
        # 对于训练，我们希望看到实时输出，所以不捕获 stdout/stderr
        self.logger.info(f"执行命令: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            self.logger.success(f"训练段 {start_iter}->{end_iter} 完成。")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"训练段失败 (返回码 {e.returncode})")

    def run_rendering(self, cmd: List[str], iteration: Optional[int] = None):
        """运行渲染脚本 (render.py)"""
        if not self.config.post_processing.enable_render:
            return

        log_msg = f"渲染 (迭代: {iteration})" if iteration else "最终渲染"
        self.logger.info(f"开始 {log_msg}...")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.success(f"{log_msg} 完成。")
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"{log_msg} 失败 (返回码 {e.returncode}): {e.stderr.strip()}"
            )
            # 不抛出异常，允许流程继续

    def run_metrics(self, cmd: List[str], iteration: Optional[int] = None):
        """运行评估脚本 (metrics.py)"""
        if not self.config.post_processing.enable_metrics:
            return

        log_msg = f"评估 (迭代: {iteration})" if iteration else "最终评估"
        self.logger.info(f"开始 {log_msg}...")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.success(f"{log_msg} 完成。")
            if result.stdout.strip():
                self.logger.info(f"评估结果:\n{result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"{log_msg} 失败 (返回码 {e.returncode}): {e.stderr.strip()}"
            )
            # 不抛出异常，允许流程继续
