from typing import List
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    iterations: int = 30000
    eval: bool = True
    quiet: bool = False
    disable_viewer: bool = True
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    densify_until_iter: int = 15000
    opacity_reset_interval: int = 3000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    opacity_lr: float = 0.05
    feature_lr: float = 0.0025
    resolution: int = -1
    lambda_dssim: float = 0.2
    sh_degree: int = 3
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    white_background: bool = False
    camera_extent: float = 1.0
    data_device: str = "cuda"
    optimizer_type: str = "default"
    test_iterations: List[int] = field(default_factory=lambda: [7000, 30000])
    save_iterations: List[int] = field(default_factory=lambda: [7000, 30000])
    checkpoint_iterations: List[int] = field(
        default_factory=lambda: []
    )  # 用于保存 .pth 检查点文件
    # 继续训练相关配置
    resume_training: bool = False
    resume_from_iteration: int = -1  # -1 表示自动检测最新的检查点


@dataclass
class PathsConfig:
    source_path: str
    model_path: str


@dataclass
class PostProcessingConfig:
    enable_render: bool = True
    enable_metrics: bool = True
    enable_intermediate_eval: bool = True


@dataclass
class SetupConfig:
    install_accelerated_rasterizer: bool = False
    rasterizer_branch: str = "3dgs_accel"


@dataclass
class AppConfig:
    """应用主配置"""

    name: str
    description: str
    paths: PathsConfig
    training: TrainingConfig
    post_processing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
