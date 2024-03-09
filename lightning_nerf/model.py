from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from typing_extensions import Literal

import torch
import nerfacc
import trimesh
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
from rich.console import Console
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, meters, misc

from .utils import FocalLoss
from .field import LightningField
from .sampler import LightningNeRFSampler

CONSOLE = Console(width=120)


@dataclass
class LightningNeRFModelConfig(ModelConfig):
    """
    LightningNeRF Model configuration.
    """

    _target: Type = field(default_factory=lambda: LightningNeRFModel)

    near_plane: float = 0.01
    """How far along the ray to start sampling."""

    far_plane: float = 6.0
    """How far along the ray to stop sampling."""

    vi_mlp_num_layers: int = 3
    """Number of layers for the view-independent MLP."""

    vi_mlp_hidden_size: int = 64
    """Hidden size for the view-independent MLP."""
    
    vd_mlp_num_layers: int = 2
    """Number of layers for the view-dependent MLP."""

    vd_mlp_hidden_size: int = 32
    """Hidden size for the view-dependent MLP."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "black", "white"] = "random"
    """The background color as RGB."""
    
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    
    point_cloud_path: Optional[Path] = None
    """Path to point cloud for initialization."""
    
    frontal_axis: Literal["x", "y"] = "x"
    """Frontal axis for the scene bounding box. This is used for createing background augmentation points."""
    
    init_density_value: float = 10.0
    """Initial density value for occupied regions in the density grid."""
    
    density_grid_base_res: int = 256
    """Base resolution for density grid."""
    
    density_log2_hashmap_size: int = 24
    """Log2 of the hashmap size for the density grid. The density grid is effectively a dense grid not a hash grid."""
    
    color_grid_base_res: int = 128
    """Base resolution for the color grid."""
    
    color_grid_max_res: int = 2048
    """Maximum resolution for the color grid."""
    
    color_grid_fpl: int = 2
    """Number of features per level for the color grid."""
    
    color_log2_hashmap_size: int = 19
    """Log2 of the hashmap size for the color grid."""
    
    color_grid_num_levels: int = 8
    """Number of (multi-scale) levels for the color grid."""
    
    bg_density_grid_res: int = 32
    """Resolution for the background density grid."""
    
    bg_density_log2_hashmap_size: int = 18
    """Log2 of the hashmap size for the background density grid."""
    
    bg_color_grid_base_res: int = 32
    """Base resolution for the background color grid."""
    
    bg_color_grid_max_res: int = 128
    """Max resolution for the background color grid."""
    
    bg_color_log2_hashmap_size: int = 16
    """Log2 of the hashmap size for the background color grid."""
    
    occ_grid_base_res: int = 256
    """Base resolution for the occupancy grid."""
    
    occ_grid_num_levels: int = 2
    """Number of (multi-scale) levels for the occupancy grid."""
    
    occ_grid_update_warmup_step: int = 2
    """Number of initial iterations for updating the occupancy grid."""
    
    occ_num_samples_per_ray: int = 1000
    """Number of samples per ray to generate for the occupancy test."""

    pdf_num_samples_per_ray: int = 8
    """Number of samples per ray after pdf samples."""
    
    pdf_samples_warmup_step: int = 500
    """Number of initial iterations for keeping the max pdf samples."""
    
    pdf_samples_fixed_step: int = 2000
    """Number of iterations for finally halving the max pdf samples."""
    
    pdf_samples_fixed_ratio: float = 0.5
    """Ratio of the max pdf samples after fixed step."""
    
    rgb_padding: Optional[float] = None
    """Padding for RGB loss. If None, no padding."""
    
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "res_rgb_loss": 0.01,
        }
    )
    """Loss coefficients."""


class LightningNeRFModel(Model):
    config: LightningNeRFModelConfig
    """LightningNeRF, Both Density and Color using tcnn. BG inverse warp. Color decomposition.

    Args:
        config: LightningNeRF configuration to instantiate model
    """
    
    def _load_pointcloud(self):
        """Load point cloud for LiDAR initialization. This is called after `populate_modules()`."""
        if self.config.point_cloud_path is None:
            CONSOLE.log(f'[blue]Point cloud path not specified.')
            return None
        
        # 1. Load point cloud
        file = trimesh.load(self.config.point_cloud_path)
        vertices = np.array(file.vertices, dtype=np.float32)

        vertices = torch.from_numpy(vertices) # (N, 3)
        
        # 2. Transform point cloud into normalized space
        vertices = (
            torch.cat(
                (
                    vertices,
                    torch.ones_like(vertices[..., :1]),
                ),
                -1,
            )
            @ self.kwargs["metadata"]["trans"].T
        )
        vertices *= self.kwargs["metadata"]["scale"]
        CONSOLE.log(f'[blue]Load vertices: {vertices.shape}')
        CONSOLE.log((vertices[:, 0]).min(), (vertices[:, 0]).max())
        CONSOLE.log((vertices[:, 1]).min(), (vertices[:, 1]).max())
        CONSOLE.log((vertices[:, 2]).min(), (vertices[:, 2]).max())
        return vertices

    def _augment_bg_points(self):
        """Augment background points on front, left, right and top surfaces of background box."""
        x_min, y_min, z_min, x_max, y_max, z_max = self.occupancy_grid.aabbs.cpu()[-1]
        x = torch.linspace(x_min, x_max, self.config.occ_grid_base_res)
        y = torch.linspace(y_min, y_max, self.config.occ_grid_base_res)
        z = torch.linspace(z_min, z_max, self.config.occ_grid_base_res)
        
        if self.config.frontal_axis == "y":
            CONSOLE.log(f'[blue]Augmenting bg points with `y` as frontal axis.')
            yy, zz = torch.meshgrid(y, z, indexing="ij")
            yy, zz = yy.flatten(), zz.flatten()
            xx_neg = torch.ones_like(yy) * x_min
            xx_pos = torch.ones_like(yy) * x_max
            aug_neg1 = torch.stack((xx_neg + 0.05, yy + 0.05, zz), -1)
            aug_pos1 = torch.stack((xx_pos - 0.05, yy - 0.05, zz), -1)
            
            xx, zz = torch.meshgrid(x, z, indexing="ij")
            xx, zz = xx.flatten(), zz.flatten()
            yy_pos = torch.ones_like(xx) * y_max
            aug_xz = torch.stack((xx, yy_pos - 0.05, zz), -1)
            
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            xx, yy = xx.flatten(), yy.flatten()
            zz_pos = torch.ones_like(xx) * z_max
            aug_xy = torch.stack((xx, yy, zz_pos - 0.05), -1)
            aug = torch.cat((aug_neg1, aug_pos1, aug_xz, aug_xy), 0)
        elif self.config.frontal_axis == "x":
            CONSOLE.log(f'[blue]Augmenting bg points with `x` as frontal axis.')
            xx, zz = torch.meshgrid(x, z, indexing="ij")
            xx, zz = xx.flatten(), zz.flatten()
            yy_neg = torch.ones_like(xx) * y_min
            yy_pos = torch.ones_like(xx) * y_max
            aug_neg1 = torch.stack((xx + 0.05, yy_neg + 0.20, zz), -1)
            aug_pos1 = torch.stack((xx - 0.05, yy_pos - 0.20, zz), -1)
            
            yy, zz = torch.meshgrid(y, z, indexing="ij")
            yy, zz = yy.flatten(), zz.flatten()
            xx_pos = torch.ones_like(yy) * x_max
            aug_yz = torch.stack((xx_pos - 0.05, yy, zz), -1)
            
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            xx, yy = xx.flatten(), yy.flatten()
            zz_pos = torch.ones_like(xx) * z_max
            aug_xy = torch.stack((xx, yy, zz_pos - 0.05), -1)
            aug = torch.cat((aug_neg1, aug_pos1, aug_yz, aug_xy), 0)
        else:
            raise ValueError(f'Invalid frontal axis: {self.config.frontal_axis}. Currently only supports "x" and "y".')
        
        CONSOLE.log(f'augment bg points: {aug.shape}')
        CONSOLE.log(f'occ grid:\n{self.occupancy_grid.aabbs}')
        return aug
        
    def _pretrain_density_grid(self):
        self.train()
        
        device = self.device
        
        vertices_fg = self.pretrain_fg_vertices
        
        CONSOLE.log(f'[blue]FG vertices: {vertices_fg.shape}')
        CONSOLE.log(vertices_fg[:,0].min(), vertices_fg[:,0].max())
        CONSOLE.log(vertices_fg[:,1].min(), vertices_fg[:,1].max())
        CONSOLE.log(vertices_fg[:,2].min(), vertices_fg[:,2].max())
        
        # Initialize density field given the pointcloud
        loss_meter = meters.AverageMeter()
        optimizer = torch.optim.Adam(self.field.density_encoding.parameters(), lr=0.2)
        
        CONSOLE.log(f'[blue]Pretraining density grid (FG) ...')
        tracker = tqdm(range(20), desc="Pretraining density grid ...")
        
        for i in tracker:
            vv = torch.tensor_split(vertices_fg, 50)
            for j in range(len(vv)):
                v = vv[j].to(device)
                query_density = self.field.density_fn(v).squeeze() # (N, )
                loss = torch.mean(torch.square(query_density - self.config.init_density_value))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())
                if j % 10 == 0:
                    tracker.set_description(f"mse loss: {loss_meter.avg:.4f}")
            self.occupancy_grid._update(i, occ_eval_fn=lambda x: self.field.density_fn(x) * self.render_step_size)
        CONSOLE.log(f'[blue]Pretrained MSE loss: {loss_meter.avg}')
        
        vertices = torch.cat((self.pretrain_bg_vertices, self.pretrain_aug_vertices), 0)
        optimizer = torch.optim.Adam(self.field.bg_density_encoding.parameters(), lr=0.2)
        
        CONSOLE.log(f'[blue]Pretraining density grid (BG) ...')
        tracker = tqdm(range(40), desc="Pretraining density grid (bg) ...")
        
        for i in tracker:
            vv = torch.tensor_split(vertices, 10)
            for j in range(len(vv)):
                v = vv[j].to(device)
                query_density = self.field.density_fn(v).squeeze() # (N, )
                loss = torch.mean(torch.square(query_density - self.config.init_density_value * 0.8))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())
                if j % 10 == 0:
                    tracker.set_description(f"mse loss: {loss_meter.avg:.4f}")
            self.occupancy_grid._update(i, occ_eval_fn=lambda x: self.field.density_fn(x) * self.render_step_size)
        CONSOLE.log(f'[blue]Pretrained MSE loss (BG): {loss_meter.avg}')
        
        self.pretrain_fg_vertices = None
        self.pretrain_bg_vertices = None
        self.pretrain_aug_vertices = None

    def _init_sampler(self):
        # auto step size: ~occ_num_samples_per_ray samples in the base level grid
        scene_aabb = self.scene_box.aabb.flatten()
        render_step_size = ((scene_aabb[3:] - scene_aabb[:3]) ** 2).sum().sqrt().item()
        self.render_step_size = render_step_size / self.config.occ_num_samples_per_ray
        CONSOLE.log(f'[blue]Render step size: {self.render_step_size}.')
        
        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=scene_aabb,
            resolution=self.config.occ_grid_base_res,
            levels=self.config.occ_grid_num_levels,
        )
        
        # Sampler
        self.sampler = LightningNeRFSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )
        
        CONSOLE.log(f'[blue]Sampler: {self.sampler}.')
        
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        will_initialize = False
        if (
            f"{prefix}field.density_encoding.params" in state_dict
            and f"{prefix}field.bg_density_encoding.params" in state_dict
            and f"{prefix}field.color_encoding.params" in state_dict
            and f"{prefix}field.bg_color_encoding.params" in state_dict
        ):
            will_initialize = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if will_initialize:
            self.grid_initialized = True
        
    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.grid_initialized = False
        
        # Fields
        CONSOLE.log(f"[blue]Scene box:\n"
                    f"  {self.scene_box.aabb[0].tolist()}\n  {self.scene_box.aabb[1].tolist()}.")
        density_grid_size = (self.scene_box.aabb[1] - self.scene_box.aabb[0]) / self.config.density_grid_base_res
        density_grid_size /= self.kwargs["metadata"]["scale"]
        CONSOLE.log(f'[blue]Density grid size (m): {density_grid_size}.')
        color_grid_size = (self.scene_box.aabb[1] - self.scene_box.aabb[0]) / self.config.color_grid_max_res
        color_grid_size /= self.kwargs["metadata"]["scale"]
        CONSOLE.log(f'[blue]Max-Res color grid size (m): {color_grid_size}.')
        
        self.field = LightningField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            density_res=self.config.density_grid_base_res,
            bg_density_res=self.config.bg_density_grid_res,
            bg_density_log2_hashmap_size=self.config.bg_density_log2_hashmap_size,
            color_base_res=self.config.color_grid_base_res,
            bg_color_base_res=self.config.bg_color_grid_base_res,
            color_max_res=self.config.color_grid_max_res,
            bg_color_max_res=self.config.bg_color_grid_max_res,
            bg_color_log2_hashmap_size=self.config.bg_color_log2_hashmap_size,
            num_levels=self.config.color_grid_num_levels,
            features_per_level=self.config.color_grid_fpl,
            density_log2_hashmap_size=self.config.density_log2_hashmap_size,
            color_log2_hashmap_size=self.config.color_log2_hashmap_size,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            vi_mlp_num_layers=self.config.vi_mlp_num_layers,
            vi_mlp_hidden_size=self.config.vi_mlp_hidden_size,
            vd_mlp_num_layers=self.config.vd_mlp_num_layers,
            vd_mlp_hidden_size=self.config.vd_mlp_hidden_size,
            rgb_padding=self.config.rgb_padding
        )
        
        # Initialize occupancy grid and sampler
        self._init_sampler()
        
        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        CONSOLE.log(f'[blue]Collider: {self.collider.near_plane}, {self.collider.far_plane}.')

        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))
        
        # losses
        self.rgb_loss = FocalLoss(alpha=1.0, type="mse")
        
        if self.config.background_color == "black":
            bg_color = torch.tensor([0., 0., 0.])
            self.register_buffer("bg_color", bg_color, persistent=False)
        elif self.config.background_color == "white":
            bg_color = torch.tensor([1., 1., 1.])
            self.register_buffer("bg_color", bg_color, persistent=False)
        else:
            # random
            self.bg_color = None
        
        if self.training:
            # Only need to load point cloud for training
            pretrain_vertices = self._load_pointcloud()
            assert pretrain_vertices is not None, "Point cloud is None."
            # Crop the point cloud into the scene bounding box
            mask = (pretrain_vertices[:,0] > self.scene_box.aabb[0][0]) & (pretrain_vertices[:,0] < self.scene_box.aabb[1][0])
            mask &= (pretrain_vertices[:,1] > self.scene_box.aabb[0][1]) & (pretrain_vertices[:,1] < self.scene_box.aabb[1][1])
            mask &= (pretrain_vertices[:,2] > self.scene_box.aabb[0][2]) & (pretrain_vertices[:,2] < self.scene_box.aabb[1][2])
            CONSOLE.log(f'[blue]#Point cloud in FG bbox: {mask.sum()}, ratio: {mask.sum()/len(mask)}.')
            self.pretrain_fg_vertices = pretrain_vertices[mask]
            self.pretrain_bg_vertices = pretrain_vertices[~mask]
            
            self.pretrain_aug_vertices = self._augment_bg_points()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        
    # Just to allow for size reduction of the checkpoint
    def load_state_dict(self, state_dict, strict: bool = True):
        for k, v in self.lpips.state_dict().items():
            state_dict[f"lpips.{k}"] = v
        return super().load_state_dict(state_dict, strict)

    # Just to allow for size reduction of the checkpoint
    def state_dict(self, *args, prefix="", **kwargs):
        state_dict = super().state_dict(*args, prefix=prefix, **kwargs)
        for k in list(state_dict.keys()):
            if k.startswith(f"{prefix}lpips."):
                state_dict.pop(k)
        return state_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        den_encoder = list()
        col_encoder = list()
        network = list()
        for name, p in self.field.named_parameters():
            if "density_encoding" in name:
                CONSOLE.log(f'[blue]density encoder: {name}, {p.shape}')
                den_encoder.append(p)
            elif "color_encoding" in name:
                CONSOLE.log(f'[blue]color encoder: {name}, {p.shape}')
                col_encoder.append(p)
            else:
                CONSOLE.log(f'[blue]network: {name}, {p.shape}')
                network.append(p)
        param_groups["den_encoder"] = den_encoder
        param_groups["col_encoder"] = col_encoder
        param_groups["network"] = network
        return param_groups

    def set_training_step(self, step):
        self.training_step += 1

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x) * self.render_step_size,
                occ_thre=self.config.alpha_thre,
                warmup_steps=self.config.occ_grid_update_warmup_step,
            )
        
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.set_training_step,
            )
        )
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            )
        )
        
        # invoke here to initialize the grid before training starts
        if not self.grid_initialized:
            self._pretrain_density_grid()
            self.grid_initialized = True

        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        num_rays = len(ray_bundle)
        
        with torch.no_grad():
            if self.training_step < self.config.pdf_samples_warmup_step:
                num_fine_samples = self.config.pdf_num_samples_per_ray
            elif self.training_step < self.config.pdf_samples_fixed_step:
                ss = self.config.pdf_num_samples_per_ray
                fixed_step = self.config.pdf_samples_fixed_step
                warmup_step = self.config.pdf_samples_warmup_step
                max_ratio = 1. - self.config.pdf_samples_fixed_ratio
                ratio = (self.training_step.item() - warmup_step) / (fixed_step - warmup_step)
                num_fine_samples = round(ss * (1 - max_ratio * ratio))
            else:
                num_fine_samples = round(self.config.pdf_num_samples_per_ray * self.config.pdf_samples_fixed_ratio)
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
                render_step_size=self.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
                num_fine_samples=num_fine_samples,
            )
        
        field_outputs = self.field(ray_samples)
        
        sigmas = field_outputs[FieldHeadNames.DENSITY]
        colors = field_outputs[FieldHeadNames.RGB]
        
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights, _, _ = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=sigmas[..., 0],
            packed_info=packed_info,
        )
        if not self.training:
            colors = torch.nan_to_num(colors)
        comp_rgb = nerfacc.accumulate_along_rays(
            weights, values=colors, ray_indices=ray_indices, n_rays=num_rays
        )
        accumulation = nerfacc.accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=num_rays
        )
        bg_color = torch.rand_like(comp_rgb) if self.bg_color is None else self.bg_color
        rgb = comp_rgb + bg_color * (1.0 - accumulation)
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        
        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        depth = nerfacc.accumulate_along_rays(
            weights, values=steps, ray_indices=ray_indices, n_rays=num_rays
        )
        depth = depth / (accumulation + 1e-10)
        depth = torch.clip(depth, steps.min(), steps.max())
        
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
        }
        
        if self.training:
            res_rgb = field_outputs.get("res_rgb", None)
            if res_rgb is not None:
                res_rgb_loss = torch.mean(res_rgb)  # for rgb combined before sigmoid
                outputs.update({"res_rgb_loss": res_rgb_loss})

        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        loss_dict = {"rgb_loss": self.rgb_loss(image, outputs["rgb"])}
        # regularizer for view-dependent color
        if outputs.get("res_rgb_loss") is not None:
            loss_dict.update({"res_rgb_loss": outputs["res_rgb_loss"]})
        
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])
        
        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]
        
        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(image, rgb).item()),
            "ssim": float(self.ssim(image, rgb)),
            "lpips": float(self.lpips(image, rgb)),
        }
        images_dict = {
            "img": combined_rgb, 
            "accumulation": combined_acc, 
            "depth": combined_depth
        }

        return metrics_dict, images_dict
