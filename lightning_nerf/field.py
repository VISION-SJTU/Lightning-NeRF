from typing import Dict, Tuple, Optional

import torch
import numpy as np
import tinycudann as tcnn
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames

CONSOLE = Console(width=120)


class LightningField(Field):
    """
    Field for Lightning NeRF.
    """
    
    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        density_res: int = 256,
        color_base_res: int = 16,
        color_max_res: int = 256,
        num_levels: int = 8,
        features_per_level: int = 2,
        density_log2_hashmap_size: int = 24,
        color_log2_hashmap_size: int = 19,
        appearance_embedding_dim: int = 0,
        bg_density_res: int = 32,
        bg_density_log2_hashmap_size: int = 18,
        bg_color_base_res: int = 32,
        bg_color_max_res: int = 128,
        bg_color_log2_hashmap_size: int = 16,
        use_average_appearance_embedding: bool = True,
        vi_mlp_num_layers: int = 3,
        vi_mlp_hidden_size: int = 64,
        vd_mlp_num_layers: int = 2,
        vd_mlp_hidden_size: int = 32,
        rgb_padding: Optional[float] = None,
    ) -> None:
        super().__init__()
        
        self.register_buffer('aabb', aabb)
        self.num_images = num_images
        self.rgb_padding = rgb_padding
        
        # Foreground density encoding, 3D -> 1D
        self.density_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 1,
                "n_features_per_level": 1,
                "log2_hashmap_size": density_log2_hashmap_size,
                "base_resolution": density_res,
                "interpolation": "Linear",
            }
        )
        # Since we use trunc_exp, we need to make sure the initialized density to be close to 0
        # exp(-3.) = 0.0498, exp(-2.8) = 0.0608
        self.density_encoding.params.data.uniform_(-3., -2.8)
        # Background density encoding, 4D -> 1D
        self.bg_density_encoding = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 1,
                "n_features_per_level": 1,
                "log2_hashmap_size": bg_density_log2_hashmap_size,
                "base_resolution": bg_density_res,
                "interpolation": "Linear",
            }
        )
        self.bg_density_encoding.params.data.uniform_(-3., -2.8)
        
        growth_factor = np.exp((np.log(color_max_res) - np.log(color_base_res)) / (num_levels - 1))
        # Foreground color encoding
        self.color_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": color_log2_hashmap_size,
                "base_resolution": color_base_res,
                "per_level_scale": growth_factor,
            }
        )
        bg_growth_factor = np.exp((np.log(bg_color_max_res) - np.log(bg_color_base_res)) / (num_levels - 1))
        # Background color encoding
        self.bg_color_encoding = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": bg_color_log2_hashmap_size,
                "base_resolution": bg_color_base_res,
                "per_level_scale": bg_growth_factor,
            }
        )
        
        # Init appearance code-related parameters
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            assert self.num_images is not None, "'num_images' must not be None when using appearance embedding"
            self.appearance_embedding = Embedding(self.num_images, self.appearance_embedding_dim)
            self.use_average_appearance_embedding = use_average_appearance_embedding  # for test-time
        
        in_dim_color = num_levels * features_per_level
        in_dim_color += self.appearance_embedding_dim
        
        # View-independent mlp does not take direction encoding
        self.vi_mlp = tcnn.Network(
            n_input_dims=in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": vi_mlp_hidden_size,
                "n_hidden_layers": vi_mlp_num_layers - 1,
            },
        )
        
        in_dim_view = num_levels * features_per_level
        in_dim_view += self.appearance_embedding_dim
        
        # View-dependent mlp takes direction encoding
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 4},
        )
        in_dim_view += self.direction_encoding.n_output_dims
        
        # View-dependent mlp takes additional direction encoding
        self.vd_mlp = tcnn.Network(
            n_input_dims=in_dim_view,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": vd_mlp_hidden_size,
                "n_hidden_layers": vd_mlp_num_layers - 1,
            },
        )
        
        CONSOLE.log(f'[blue]vi mlp input: {self.vi_mlp.n_input_dims}, vd mlp input: {self.vd_mlp.n_input_dims}')
    
    def density_fn(self, positions: TensorType) -> TensorType:
        positions_flat = positions.view(-1, 3)
        
        # Get nomrlaized positions for tcnn, range [0, 1]
        positions_norm = SceneBox.get_normalized_positions(positions_flat, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions_norm > 0.0) & (positions_norm < 1.0)).all(dim=-1)
        
        if selector.all():
            density_before_act = self.density_encoding(positions_norm)
        else:
            fg_selector = torch.where(selector)[0]
            fg_positions_norm = positions_norm[fg_selector]
            fg_density_before_act = self.density_encoding(fg_positions_norm)
            
            # There are some positions outside the box
            bg_selector = torch.where(~selector)[0]
            bg_positions_norm = positions_norm[bg_selector]
            # bg is considered as points outside unit box (centered at 0, with length 1)
            # so we need to shift [0, 1] to [-1, 1]
            bg_positions_norm = bg_positions_norm * 2.0 - 1.0
            # inf norm of bg positions
            bg_inf_norm = bg_positions_norm.abs().max(-1, keepdim=True).values
            # shift [-1, 1] to [0, 1] for tcnn
            bg_positions_warp = (bg_positions_norm / bg_inf_norm + 1.0) / 2.0
            bg_positions_warp = torch.cat([bg_positions_warp, torch.div(1., bg_inf_norm)], dim=-1)
            bg_density_before_act = self.bg_density_encoding(bg_positions_warp)
        
            density_before_act = torch.zeros(
                (positions_flat.shape[0], 1), 
                device=fg_density_before_act.device,
                dtype=fg_density_before_act.dtype
            )
            density_before_act[fg_selector] = fg_density_before_act
            density_before_act[bg_selector] = bg_density_before_act

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_act.to(positions))
        density = density.view(*positions.shape[:-1], 1)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, 3)
        
        # Get nomrlaized positions for tcnn, range [0, 1]
        positions_norm = SceneBox.get_normalized_positions(positions_flat, self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions_norm > 0.0) & (positions_norm < 1.0)).all(dim=-1)
        
        if selector.all():
            density_before_act = self.density_encoding(positions_norm)
            color_embedding = self.color_encoding(positions_norm)
        else:
            fg_selector = torch.where(selector)[0]
            fg_positions_norm = positions_norm[fg_selector]
            fg_density_before_act = self.density_encoding(fg_positions_norm)
            fg_color_embedding = self.color_encoding(fg_positions_norm)

            # There are some positions outside the box
            bg_selector = torch.where(~selector)[0]
            bg_positions_norm = positions_norm[bg_selector]
            # bg is considered as points outside unit box (centered at 0, with length 1)
            # so we need to shift [0, 1] to [-1, 1]
            bg_positions_norm = bg_positions_norm * 2.0 - 1.0 
            # inf norm of bg positions
            bg_inf_norm = bg_positions_norm.abs().max(-1, keepdim=True).values
            # shift [-1, 1] to [0, 1] for tcnn
            bg_positions_warp = (bg_positions_norm / bg_inf_norm + 1.0) / 2.0
            bg_positions_warp = torch.cat([bg_positions_warp, torch.div(1., bg_inf_norm)], dim=-1)
            bg_density_before_act = self.bg_density_encoding(bg_positions_warp)
            bg_color_embedding = self.bg_color_encoding(bg_positions_warp)
        
            density_before_act = torch.zeros(
                (positions_flat.shape[0], 1), 
                device=fg_density_before_act.device,
                dtype=fg_density_before_act.dtype
            )
            density_before_act[fg_selector] = fg_density_before_act
            density_before_act[bg_selector] = bg_density_before_act
            
            color_embedding = torch.zeros(
                (positions_flat.shape[0], self.color_encoding.n_output_dims), 
                device=fg_color_embedding.device,
                dtype=fg_color_embedding.dtype
            )
            color_embedding[fg_selector] = fg_color_embedding
            color_embedding[bg_selector] = bg_color_embedding

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_act.to(positions))
        density = density.view(*positions.shape[:-1], 1)
        color_embedding = color_embedding.view(*positions.shape[:-1], -1)
        return density, color_embedding
        
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        output_shape = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions.reshape(-1, 3)

        color_features = [density_embedding.view(-1, self.color_encoding.n_output_dims)]
        view_features = [density_embedding.view(-1, self.color_encoding.n_output_dims)]
        
        if self.appearance_embedding_dim > 0:
            if self.training:
                assert ray_samples.camera_indices is not None, "'camera_indices' must not be None when training"
                camera_indices = ray_samples.camera_indices.squeeze()
                appearance_embedding = self.appearance_embedding(camera_indices)
            elif self.use_average_appearance_embedding:
                appearance_embedding = torch.ones(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.mean(dim=0)
            else:
                appearance_embedding = torch.zeros(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                )
        
            appearance_embedding = appearance_embedding.view(-1, self.appearance_embedding_dim)
            color_features.append(appearance_embedding)
            view_features.append(appearance_embedding)
        
        color_features = torch.cat(color_features, dim=-1)

        view_features.append(self.direction_encoding(directions))
        view_features = torch.cat(view_features, dim=-1)
        
        vi_rgb = self.vi_mlp(color_features).view(*output_shape, -1).to(directions)
        vd_rgb = self.vd_mlp(view_features).view(*output_shape, -1).to(directions)
        rgb = vi_rgb + vd_rgb   # NOTE: no sigmoid as each already in (0,1)
        
        if self.rgb_padding is not None:
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        
        return {FieldHeadNames.RGB: rgb, "res_rgb": vd_rgb}
