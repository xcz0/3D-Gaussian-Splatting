#!/usr/bin/env python3
"""
工具函数模块
"""

import os
import time
import yaml
import shutil
import subprocess
from pathlib import Path
from contextlib import contextmanager
from typing import List

from config import (
    AppConfig,
    PathsConfig,
    TrainingConfig,
    PostProcessingConfig,
    SetupConfig,
)
from logger import Logger


def load_config(config_path: Path) -> AppConfig:
    """加载并验证 YAML 配置文件"""
    logger = Logger("ConfigLoader")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 简单的必需字段验证
        for key in ["name", "paths", "training"]:
            if key not in data:
                raise ValueError(f"配置文件缺少必需字段: {key}")

        # 使用字典解包填充 dataclass
        return AppConfig(
            name=data["name"],
            description=data.get("description", "无描述"),
            paths=PathsConfig(**data["paths"]),
            training=TrainingConfig(**data.get("training", {})),
            post_processing=PostProcessingConfig(**data.get("post_processing", {})),
            setup=SetupConfig(**data.get("setup", {})),
        )
    except (FileNotFoundError, yaml.YAMLError, TypeError, ValueError) as e:
        logger.error(f"加载配置文件 {config_path} 失败: {e}")
        raise


def validate_paths(config: AppConfig, gaussian_splatting_dir: Path):
    """验证所有必要的路径"""
    logger = Logger("PathValidator")

    if not gaussian_splatting_dir.exists():
        raise FileNotFoundError(f"核心目录不存在: {gaussian_splatting_dir}")

    source_path = Path(config.paths.source_path)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"源数据路径不存在: {source_path}")

    # 确保输出目录存在
    Path(config.paths.model_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("所有路径验证通过。")


@contextmanager
def chdir_context(path: Path):
    """安全切换工作目录的上下文管理器"""
    logger = Logger("DirectoryChanger")
    original_cwd = os.getcwd()
    try:
        logger.info(f"切换工作目录到: {path}")
        os.chdir(path)
        yield
    finally:
        logger.info(f"恢复工作目录: {original_cwd}")
        os.chdir(original_cwd)


def setup_environment(config: AppConfig, gaussian_splatting_dir: Path):
    """设置 Python 路径和加速器"""
    logger = Logger("EnvironmentSetup")

    # 设置 PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(gaussian_splatting_dir) not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{gaussian_splatting_dir}{os.pathsep}{pythonpath}"
        logger.info("已更新 PYTHONPATH")

    # 安装加速器
    if config.setup.install_accelerated_rasterizer:
        install_accelerated_rasterizer(config, gaussian_splatting_dir)


def install_accelerated_rasterizer(config: AppConfig, gaussian_splatting_dir: Path):
    """安装或更新加速版的 rasterizer"""
    logger = Logger("RasterizerInstaller")
    logger.info("正在尝试安装加速版 rasterizer...")

    rasterizer_dir = gaussian_splatting_dir / "submodules/diff-gaussian-rasterization"
    if not rasterizer_dir.exists():
        logger.warning(f"Rasterizer 目录不存在: {rasterizer_dir}，跳过安装。")
        return

    with chdir_context(rasterizer_dir):
        try:
            # 卸载旧版，切换分支，安装新版
            run_command(
                ["pip", "uninstall", "diff-gaussian-rasterization", "-y"],
                "卸载旧版 rasterizer",
            )
            if (build_dir := rasterizer_dir / "build").exists():
                shutil.rmtree(build_dir)
            run_command(
                ["git", "checkout", config.setup.rasterizer_branch],
                "切换 git 分支",
            )
            run_command(["pip", "install", "."], "安装新版 rasterizer")
            logger.success("加速版 rasterizer 安装成功。")
        except RuntimeError as e:
            logger.error(f"安装加速版 rasterizer 失败: {e}")
            logger.warning("将继续使用默认版本的 rasterizer。")


def run_command(cmd: List[str], log_message: str) -> subprocess.CompletedProcess:
    """通用命令执行器"""
    logger = Logger("CommandRunner")
    logger.info(f"执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if result.stdout.strip():
            logger.info(f"输出:\n{result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"{log_message} 失败 (返回码: {e.returncode}).\n错误: {e.stderr.strip()}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def detect_existing_checkpoints(model_path: str) -> List[int]:
    """检测模型路径中已存在的检查点"""
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        return []

    checkpoints = []

    # 检查 .pth 检查点文件（用于继续训练）
    for checkpoint_file in model_path_obj.glob("chkpnt*.pth"):
        try:
            # 从 chkpnt<iteration>.pth 中提取迭代次数
            iteration_str = checkpoint_file.stem.replace("chkpnt", "")
            iteration = int(iteration_str)
            checkpoints.append(iteration)
        except ValueError:
            continue

    # 检查 point_cloud 目录中的检查点文件（作为备选）
    point_cloud_dir = model_path_obj / "point_cloud"
    if point_cloud_dir.exists():
        for iteration_dir in point_cloud_dir.glob("iteration_*"):
            if iteration_dir.is_dir():
                try:
                    iteration = int(iteration_dir.name.split("_")[-1])
                    checkpoints.append(iteration)
                except ValueError:
                    continue

    return sorted(checkpoints)


def get_resume_iteration(config: AppConfig) -> int:
    """确定从哪个迭代开始继续训练"""
    logger = Logger("ResumeHelper")

    if not config.training.resume_training:
        return 0

    existing_checkpoints = detect_existing_checkpoints(config.paths.model_path)
    if not existing_checkpoints:
        logger.warning("启用了继续训练但未找到现有检查点，将从头开始训练")
        return 0

    if config.training.resume_from_iteration == -1:
        # 自动选择最新的检查点
        resume_iter = max(existing_checkpoints)
        logger.info(f"自动检测到最新检查点: 第 {resume_iter} 次迭代")
    else:
        # 使用指定的迭代点
        resume_iter = config.training.resume_from_iteration
        if resume_iter not in existing_checkpoints:
            logger.warning(
                f"指定的迭代点 {resume_iter} 不存在，可用的检查点: {existing_checkpoints}"
            )
            resume_iter = max(existing_checkpoints)
            logger.info(f"使用最新的检查点: 第 {resume_iter} 次迭代")

    return resume_iter


def print_final_results(config: AppConfig, start_time: float):
    """打印最终的训练结果信息"""
    logger = Logger("ResultsPrinter")

    total_time = time.time() - start_time if start_time else 0
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info("=" * 60)
    logger.success("训练流程全部完成!")
    logger.info(f"总耗时: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    model_path = Path(config.paths.model_path).resolve()
    logger.info(f"结果保存在: {model_path}")

    if (results_file := model_path / "results.json").exists():
        logger.info(f"- 评估指标: {results_file}")
    if (test_dir := model_path / "test").exists():
        count = len(list(test_dir.glob("*.png")))
        logger.info(f"- 测试集渲染图片: {test_dir} ({count} 张)")
    logger.info("=" * 60)


def list_available_configs():
    """列出 config/ 目录下的所有可用配置文件"""
    config_dir = Path(__file__).parent / "config"
    if not config_dir.exists():
        print("配置目录 'config/' 不存在。")
        return

    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        print("在 'config/' 目录中未找到任何 .yaml 配置文件。")
        return

    print("\n可用的配置文件:")
    print("=" * 50)
    for i, path in enumerate(configs, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            print(f"{i}. {path.name}")
            print(f"   名称: {data.get('name', 'N/A')}")
            print(f"   描述: {data.get('description', '无')}")
            print(f"   迭代: {data.get('training', {}).get('iterations', 'N/A')}")
            print("-" * 20)
        except Exception as e:
            print(f"{i}. {path.name} - 读取失败: {e}")
    print("=" * 50)
    print(f"使用示例: python main.py -c config/{configs[0].name}")
