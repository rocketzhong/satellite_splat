"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from satellite_splat.satellite_splat_datamanager import (
    SatelliteSplatDataManangerConfig,
)
from satellite_splat.satellite_splat_model import SatelliteSplatModelConfig
from satellite_splat.satellite_splat_pipeline import (
    SatelliteSplatPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from satellite_splat.satellite_splat_dataparser import SatelliteDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from satellite_splat.satellite_splat_trainer import SatelliteSplatTrainerConfig
from nerfstudio.plugins.types import MethodSpecification

max_steps = 30000
satellite_splat = MethodSpecification(
    config=SatelliteSplatTrainerConfig(
        method_name="satellite-splat",  # TODO: rename to your own model
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=max_steps,
        mixed_precision=True,
        pipeline=SatelliteSplatPipelineConfig(
            datamanager=SatelliteSplatDataManangerConfig(
                dataparser=SatelliteDataParserConfig(
                    alpha_color="black"
                )
            ),
            model=SatelliteSplatModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                background_color="black",
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=max_steps,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(
            num_rays_per_chunk=1 << 15,
            websocket_port=7007
        ),
        vis="viewer_legacy",
    ),
    description="Satellite Splat following Nerfstudio method template.",
)
