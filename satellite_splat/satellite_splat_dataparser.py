from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig,Blender
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.utils.io import load_from_json


@dataclass
class SatelliteDataParserConfig(BlenderDataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: SatelliteDataParser)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = None
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
    ply_path: Optional[Path] = Path("points3d.ply")
    """Path to PLY file to load 3D points from, defined relative to the dataset directory. This is helpful for
    Gaussian splatting and generally unused otherwise. If `None`, points are initialized randomly."""


class SatelliteDataParser(Blender):
    """Overring Blender Dataset
    Blender from:
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: SatelliteDataParserConfig

    def __init__(self, config: SatelliteDataParserConfig):
        super().__init__(config=config)

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".jpg")
            image_filenames.append(fname)
            c2w = np.array(frame["transform_matrix"])
            c2w[3: 0:1] -= -1 # flip the y and z axis
            poses.append(c2w)
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor


        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {}
        if self.config.ply_path is not None:
            metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))

        pcds = metadata['points3D_xyz']
        pcd_center = np.mean(pcds, axis=0)
        x_c = pcd_center[0]
        y_c = pcd_center[1]
        z_c = pcd_center[2]
        scene_box = SceneBox(aabb=torch.tensor([[x_c-0.5, y_c-0.5, z_c-0.5], [x_c+0.5, y_c+0.5, z_c+0.5]], dtype=torch.float32))
        
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )

        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path):
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        #check if exist
        if ply_file_path.exists():
            full_ply_file_path = ply_file_path
        else:
            full_ply_file_path = self.data / ply_file_path

        if not full_ply_file_path.exists():
            raise FileNotFoundError(f"File {ply_file_path} not found.")

        pcd = o3d.io.read_point_cloud(str(full_ply_file_path))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.config.scale_factor)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out
