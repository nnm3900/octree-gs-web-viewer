from plyfile import PlyData, PlyElement
import numpy as np
from io import BytesIO
import torch
from torchsummary import summary

class Octree:
    def __init__(self, min_coord, max_coord, max_depth):
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.max_depth = max_depth
        self.children = [None] * 8
        self.indices = []

    def insert(self, point, index, depth=0):
        if depth == self.max_depth:
            self.indices.append(index)
            return

        octant = self.get_octant(point)
        if self.children[octant] is None:
            mid_coord = (self.min_coord + self.max_coord) / 2
            child_min = np.where(octant & (1 << np.arange(3)), mid_coord, self.min_coord)
            child_max = np.where(octant & (1 << np.arange(3)), self.max_coord, mid_coord)
            self.children[octant] = Octree(child_min, child_max, self.max_depth)
        self.children[octant].insert(point, index, depth + 1)

    def get_octant(self, point):
        mid_coord = (self.min_coord + self.max_coord) / 2
        octant = 0
        for i in range(3):
            if point[i] >= mid_coord[i]:
                octant |= 1 << i
        return octant

    def get_sorted_indices(self, indices):
        for child in self.children:
            if child is not None:
                child.get_sorted_indices(indices)
        indices.extend(self.indices)

def convert_ply(SourceDir, OutputDir, ply_file_path):
    plydata = PlyData.read(SourceDir + ply_file_path)
    vert = plydata["vertex"]

    min_coord = np.array([vert["x"].min(), vert["y"].min(), vert["z"].min()])
    max_coord = np.array([vert["x"].max(), vert["y"].max(), vert["z"].max()])
    octree = Octree(min_coord, max_coord, 5)  
    for i in range(len(vert)):
        point = np.array([vert["x"][i], vert["y"][i], vert["z"][i]])
        octree.insert(point, i)

    sorted_indices = []
    octree.get_sorted_indices(sorted_indices)

    anchor_pos = []
    gsplat_positions = []
    anchor_scales = []
    anchor_features = []
    anchor_levels = []
    anchor_extra_levels = []
    anchor_info = []
    offset_names = [p.name for p in vert.properties if p.name.startswith("f_offset")]
    offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
    offsets = np.zeros((len(vert["x"]), len(offset_names)))
    for idx, attr_name in enumerate(offset_names):
        offsets[:, idx] = np.asarray(vert[attr_name]).astype(np.float32)
    offsets = offsets.reshape((offsets.shape[0], 3, -1))

    for i in sorted_indices:
        anchor_pos.append(vert["x"][i])
        anchor_pos.append(vert["y"][i])
        anchor_pos.append(vert["z"][i])
        anchor_levels.append(vert["level"][i])
        anchor_extra_levels.append(vert["extra_level"][i])
        for j in range(10):
            gsplat_positions.append(vert["x"][i] + offsets[i][0][j] * np.exp(vert["scale_0"][i]))
            gsplat_positions.append(vert["y"][i] + offsets[i][1][j] * np.exp(vert["scale_1"][i]))
            gsplat_positions.append(vert["z"][i] + offsets[i][2][j] * np.exp(vert["scale_2"][i]))
        for j in range(3):
            anchor_scales.append(np.exp(vert["scale_" + str(j + 3)][i]))
        for j in range(32):
            anchor_features.append(vert["f_anchor_feat_" + str(j)][i])

    gaussians_num_per_anchor = 10
    camera_position = np.array([0, 0, 0])
    anchor_num = len(anchor_pos) // 3
    gaussians_num = anchor_num * gaussians_num_per_anchor
    print("num_gaussians: ", gaussians_num)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
         ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
         ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'), ('f_rest_3', 'f4'), ('f_rest_4', 'f4'),
         ('f_rest_5', 'f4'), ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'), ('f_rest_9', 'f4'),
         ('f_rest_10', 'f4'), ('f_rest_11', 'f4'), ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'),
         ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'), ('f_rest_18', 'f4'), ('f_rest_19', 'f4'),
         ('f_rest_20', 'f4'), ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'), ('f_rest_24', 'f4'),
         ('f_rest_25', 'f4'), ('f_rest_26', 'f4'), ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'),
         ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'), ('f_rest_33', 'f4'), ('f_rest_34', 'f4'),
         ('f_rest_35', 'f4'), ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'), ('f_rest_39', 'f4'),
         ('f_rest_40', 'f4'), ('f_rest_41', 'f4'), ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'),
         ('opacity', 'f4'),
         ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    gaussian_data = np.zeros(gaussians_num, dtype=dtype)
    gaussian_data['x'] = gsplat_positions[::3][:gaussians_num]
    gaussian_data['y'] = gsplat_positions[1::3][:gaussians_num]
    gaussian_data['z'] = gsplat_positions[2::3][:gaussians_num]

    color_mlp = torch.jit.load( SourceDir + "color_mlp.pt")
    cov_mlp = torch.jit.load( SourceDir + "cov_mlp.pt")
    opacity_mlp = torch.jit.load( SourceDir + "opacity_mlp.pt")

    anchor_features_np = np.array(anchor_features)
    anchor_pos_np = np.array(anchor_pos)
    camera_position_np = np.array(camera_position)
    anchor_feature_with_camera = np.zeros((anchor_num, 35), dtype=np.float32)
    anchor_feature_with_camera[:, :32] = anchor_features_np.reshape(anchor_num, 32)
    dx = anchor_pos_np[::3] - camera_position_np[0]
    dy = anchor_pos_np[1::3] - camera_position_np[1]  
    dz = anchor_pos_np[2::3] + camera_position_np[2]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    anchor_feature_with_camera[:, 32] = dx / distance
    anchor_feature_with_camera[:, 33] = dy / distance
    anchor_feature_with_camera[:, 34] = dz / distance

    anchor_feature_with_camera_tensor = torch.from_numpy(anchor_feature_with_camera).unsqueeze(1).unsqueeze(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_feature_with_camera_tensor = anchor_feature_with_camera_tensor.to(device)

    with torch.no_grad():
        color_output = color_mlp(anchor_feature_with_camera_tensor)
        cov_output = cov_mlp(anchor_feature_with_camera_tensor)
        opacity_output = opacity_mlp(anchor_feature_with_camera_tensor)
    color_output = color_output.squeeze().cpu().numpy()
    color_output = color_output.reshape(-1)
    cov_output = cov_output.squeeze().cpu().numpy()
    cov_output = cov_output.reshape(-1)
    opacity_output = opacity_output.squeeze().cpu().numpy()
    opacity_output = opacity_output.reshape(-1)

    SH_C0 = 0.28209479177387814
    gaussian_data['f_dc_0'] = (color_output[::3][:gaussians_num] - 0.5) / SH_C0
    gaussian_data['f_dc_1'] = (color_output[1::3][:gaussians_num] - 0.5) / SH_C0
    gaussian_data['f_dc_2'] = (color_output[2::3][:gaussians_num] - 0.5) / SH_C0
    for i in range(gaussians_num):
        if opacity_output[i] < 0:
            gaussian_data['opacity'][i] = -100
        else:
            gaussian_data['opacity'][i] = - np.log(1/opacity_output[i] - 1)
    for i in range(gaussians_num):
        gaussian_data['scale_0'][i] = np.log(sigmoid(cov_output[i * 7 + 0]) * anchor_scales[(i//gaussians_num_per_anchor) * 3 + 0])
        gaussian_data['scale_1'][i] = np.log(sigmoid(cov_output[i * 7 + 1]) * anchor_scales[(i//gaussians_num_per_anchor) * 3 + 1])
        gaussian_data['scale_2'][i] = np.log(sigmoid(cov_output[i * 7 + 2]) * anchor_scales[(i//gaussians_num_per_anchor) * 3 + 2])
    cov_vec_0 = []
    cov_vec_1 = []
    cov_vec_2 = []
    cov_vec_3 = []
    for i in range(gaussians_num):
        vec = cov_output[i * 7 + 3:i * 7 + 7]
        vec = vec / np.linalg.norm(vec)
        cov_vec_0.append(vec[0])
        cov_vec_1.append(vec[1])
        cov_vec_2.append(vec[2])
        cov_vec_3.append(vec[3])
    gaussian_data['rot_0'] = cov_vec_0
    gaussian_data['rot_1'] = cov_vec_1
    gaussian_data['rot_2'] = cov_vec_2
    gaussian_data['rot_3'] = cov_vec_3

    opacity_threshold = -100
    filtered_gaussian_data = []
    for i in range(gaussians_num):
        if opacity_output[i] > opacity_threshold:
            gaussian_data_item = tuple(gaussian_data[i])
            filtered_gaussian_data.append(gaussian_data_item)

    vertex_element = PlyElement.describe(np.array(filtered_gaussian_data, dtype=dtype), 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write('output.ply')
    plydata = PlyData.read('output.ply')
    print(f"Number of vertices in output.ply: {len(plydata['vertex'])}")
    print("Conversion complete.")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    SourceDir = "./gs-data/"
    OutputDir = "./converted/"
    convert_ply(SourceDir, OutputDir, "point_cloud.ply")

