from __future__ import annotations

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from .model import LightningNeRFModelConfig


lightning_nerf_method = MethodSpecification(
    config = TrainerConfig(
        method_name="lightning_nerf",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        steps_per_eval_image=30000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=65536,
                eval_num_rays_per_batch=2048,
            ),
            model=LightningNeRFModelConfig(
                eval_num_rays_per_chunk=1<<17,
            ),
        ),
        optimizers={
            "den_encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=10,
                    ramp="linear",
                    lr_final=0.01,
                    max_steps=10_000,
                ),
            },
            "col_encoder": {
                "optimizer": RAdamOptimizerConfig(lr=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    warmup_steps=10,
                    ramp="linear",
                    lr_final=0.01,
                    max_steps=10_000,
                ),
            },
            "network": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001,
                    max_steps=30_000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="LightningNeRF for efficient training and rendering in autonomous driving scenarios."
)
