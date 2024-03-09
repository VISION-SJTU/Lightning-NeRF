from typing import Callable, List, Optional, Tuple, Union

import torch
import nerfacc
from nerfacc import OccGridEstimator
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.model_components.ray_samplers import Sampler

from nr3d_lib.render.pack_ops import merge_two_packs_sorted_aligned
from nr3d_lib.render.raysample import packed_sample_pdf

CONSOLE = Console(width=120)


@torch.no_grad()
def packed_append(
    feat: torch.Tensor,
    other: torch.Tensor,
    num_append_per_pack: int,
    ray_indices: torch.LongTensor,
    pack_infos: torch.LongTensor
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """ Append a scaler to each packed tensor.

    Args:
        feat (torch.Tensor): Original packed features. Shape: (num_feat,)
        other (torch.Tensor): Features to append. Shape: (num_pack, num_append_per_pack)
        num_append_per_pack (int): Number of items to append to each pack.
        ray_indices (torch.LongTensor): Ray indices of the original packed features. Shape: (num_feat,)
        pack_infos (torch.LongTensor): Packed info of the original packed features. Shape: (num_pack, 2)

    Returns:
        Tuple[torch.Tensor, torch.LongTensor]: 
            Output features. Shape: (num_feat + num_pack * num_append_per_pack,)
            Output packed info. Shape: (num_pack, 2)
    """
    num_rays = pack_infos.shape[0]
    new_vals = feat.new_empty([feat.numel() + num_rays * num_append_per_pack])
    original_indices = torch.arange(feat.numel(), device=feat.device)                    # (num_feat,)
    original_aranges = torch.arange(num_rays * num_append_per_pack, device=feat.device)  # (num_rays,)
    
    pidx_a = original_indices + (ray_indices * num_append_per_pack)
    
    expand_pack_infos = pack_infos.unsqueeze(1).expand(-1, num_append_per_pack, -1).reshape(-1, 2)
    pidx_b = expand_pack_infos[:, 0] + expand_pack_infos[:, 1] + original_aranges
    
    new_vals.scatter_(0, pidx_a, feat)
    new_vals.scatter_(0, pidx_b, other)
    
    new_packed_infos: torch.LongTensor = pack_infos.clone()
    new_packed_infos[:, 0] += original_aranges[::num_append_per_pack]
    new_packed_infos[:, 1] += num_append_per_pack
    return new_vals, new_packed_infos


class LightningNeRFSampler(Sampler):
    """
    Optionally perform PDF sampling after occ grid sampling.

    Args:
    occupancy_grid: Occupancy grid to sample from.
    density_fn: Function that evaluates density at a given point.
    scene_aabb: Axis-aligned bounding box of the scene, should be set to None if the scene is unbounded.
    """

    def __init__(
        self,
        occupancy_grid: OccGridEstimator,
        density_fn: Optional[Callable[[TensorType[..., 3]], TensorType[..., 1]]] = None,
    ) -> None:

        super().__init__()
        assert occupancy_grid is not None
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid

    def get_sigma_fn(self, origins, directions) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:
            origins: Origins of rays
            directions: Directions of rays
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        # Modified here to also discard sampling points during inference for speed.
        if self.density_fn is None:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            return density_fn(positions).squeeze(-1)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The GridNeRFSampler fuses sample generation and density check together. Please call forward() directly."
        )

    # pylint: disable=arguments-differ
    def forward(
        self,
        ray_bundle: RayBundle,
        render_step_size: float,
        near_plane: float = 0.0,
        far_plane: Optional[float] = None,
        alpha_thre: float = 0.01,
        cone_angle: float = 0.0,
        num_fine_samples: int = 32,
    ) -> Tuple[RaySamples, TensorType["total_samples",]]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.
            num_fine_samples: Number of importance samples to generate per ray.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """
        # We only model static scene, so no need to fetch ray_bundle.times
        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None
        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)
        
        if num_fine_samples > 0:
            # pdf sampling here
            origins = rays_o[ray_indices]
            dirs = rays_d[ray_indices]
            
            # transform dilated ray indices to compact ray indices
            unique_ray_indices = torch.unique(ray_indices)
            # criticial api for speed
            new_ray_indices = torch.bucketize(ray_indices, unique_ray_indices, right=False)
            new_num_rays = len(unique_ray_indices)
            
            packed_info_uniform = nerfacc.pack_info(new_ray_indices, new_num_rays)
            positions_uniform = origins + dirs * (starts + ends)[:, None] / 2.0
            sigmas_uniform = self.density_fn(positions_uniform).squeeze()
            
            weights_uniform, _, _ = nerfacc.render_weight_from_density(
                t_starts=starts,
                t_ends=ends,
                sigmas=sigmas_uniform,
                packed_info=packed_info_uniform,
            )
            
            weights_uniform = weights_uniform + 1e-2
            trailing_zeros = torch.zeros(new_num_rays, dtype=weights_uniform.dtype, device=weights_uniform.device)
            # insert trailing zeros to the end of each pack
            weights_uniform, packed_info_merged = packed_append(
                weights_uniform, trailing_zeros, 1, new_ray_indices, packed_info_uniform
            )

            last_sample_id = packed_info_uniform[:, 0] + packed_info_uniform[:, 1] - 1
            trailing_ends = ends[last_sample_id]
            # insert trailing ends to the end of each pack
            existing_bins, _ = packed_append(
                starts, trailing_ends, 1, new_ray_indices, packed_info_uniform
            )

            # (num_packs, num_fine_samples + 1)
            samples_t = packed_sample_pdf(
                bins=existing_bins,
                weights=weights_uniform,
                pack_infos=packed_info_merged,
                num_to_sample=num_fine_samples + 1,
                perturb=self.training
            )[0]
            
            # combine coarse and fine samples
            # combination is importance for performance
            ray_indices_fine = torch.arange(new_num_rays, device=rays_o.device).repeat_interleave(num_fine_samples)
            packed_info_fine = nerfacc.pack_info(ray_indices_fine, new_num_rays)

            # TODO: Possible speedup here by performing merge only once
            starts = merge_two_packs_sorted_aligned(
                starts, packed_info_uniform,
                samples_t[..., :-1].reshape(-1), packed_info_fine,
                b_sorted=True, return_val=True
            )[0]
            ends = merge_two_packs_sorted_aligned(
                ends, packed_info_uniform,
                samples_t[..., 1:].reshape(-1), packed_info_fine,
                b_sorted=True, return_val=True
            )[0]
            
            ray_indices_complete = packed_append(
                new_ray_indices, ray_indices_fine, num_fine_samples, new_ray_indices, packed_info_uniform
            )[0]
            
            ray_indices = unique_ray_indices[ray_indices_complete]

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        zeros = torch.zeros_like(origins[:, :1])
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=zeros,
            ),
            camera_indices=camera_indices,
        )
        return ray_samples, ray_indices
