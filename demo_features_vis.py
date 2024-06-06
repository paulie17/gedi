import torch
import numpy as np
import open3d as o3d
from gedi import GeDi

from sklearn.decomposition import PCA

config = {'dim': 32,                                            # descriptor output dimension
          'samples_per_batch': 500,                             # batches to process the data on GPU
          'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
          'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
          'r_lrf': .5,                                          # LRF radius
          'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

voxel_size = .002
patches_per_pair = 5000

# initialising class
gedi = GeDi(config=config)

# getting a pair of point clouds
pcd0 = o3d.io.read_point_cloud('meshes/SinkSmall_meters.ply')


# randomly sampling some points from the point cloud
inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)

pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()

# applying voxelisation to the point cloud
pcd0 = pcd0.voxel_down_sample(voxel_size)

_pcd0 = torch.tensor(np.asarray(pcd0.points)).float()

# computing descriptors
pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
print(pcd0_desc.shape)

# PCA features to rgb mapping
pca = PCA(n_components=3) 
data_pca = pca.fit_transform(pcd0_desc)
# Normalize the principal components to [0, 1] for RGB mapping
min_max_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
data_pca_normalized = np.array([min_max_scaler(component) for component in data_pca.T]).T
# Use the normalized PCA components as RGB values
colors = data_pca_normalized

# preparing format for open3d ransac
pcd0_dsdv = o3d.pipelines.registration.Feature()
pcd0_dsdv.data = pcd0_desc.T

_pcd0 = o3d.geometry.PointCloud()
_pcd0.points = o3d.utility.Vector3dVector(pts0)
_pcd0.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([_pcd0])
