from plyfile import PlyData
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
    print(plydata)
    vert = plydata["vertex"]
    print("anchor num: ", len(vert))

    min_coord = np.array([vert["x"].min(), vert["y"].min(), vert["z"].min()])
    max_coord = np.array([vert["x"].max(), vert["y"].max(), vert["z"].max()])
    octree = Octree(min_coord, max_coord, 5)  
    for i in range(len(vert)):
        point = np.array([vert["x"][i], vert["y"][i], vert["z"][i]])
        octree.insert(point, i)

    sorted_indices = []
    octree.get_sorted_indices(sorted_indices)

    sorted_anchor_pos = []
    sorted_gsplat_positions = []
    sorted_anchor_scales = []
    sorted_anchor_features = []
    sorted_anchor_levels = []
    sorted_anchor_extra_levels = []
    anchor_info = []

    offset_names = [p.name for p in vert.properties if p.name.startswith("f_offset")]
    offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
    offsets = np.zeros((len(vert["x"]), len(offset_names)))
    for idx, attr_name in enumerate(offset_names):
        offsets[:, idx] = np.asarray(vert[attr_name]).astype(np.float32)
    offsets = offsets.reshape((offsets.shape[0], 3, -1))

    for i in sorted_indices:
        sorted_anchor_pos.append(vert["x"][i])
        sorted_anchor_pos.append(vert["y"][i])
        sorted_anchor_pos.append(vert["z"][i])
        sorted_anchor_levels.append(vert["level"][i])
        sorted_anchor_extra_levels.append(vert["extra_level"][i])
        for j in range(10):
            sorted_gsplat_positions.append(vert["x"][i] + offsets[i][0][j] * np.exp(vert["scale_0"][i]))
            sorted_gsplat_positions.append(vert["y"][i] + offsets[i][1][j] * np.exp(vert["scale_1"][i]))
            sorted_gsplat_positions.append(vert["z"][i] + offsets[i][2][j] * np.exp(vert["scale_2"][i]))
        for j in range(3):
            sorted_anchor_scales.append(np.exp(vert["scale_" + str(j + 3)][i]))
        for j in range(32):
            sorted_anchor_features.append(vert["f_anchor_feat_" + str(j)][i])

    anchor_info.append(vert["info"][0])
    anchor_info.append(vert["info"][1])
    
    count_elements(sorted_anchor_levels)
    count_elements(sorted_anchor_extra_levels)

    save_bin(sorted_anchor_pos, OutputDir + "anchor_positions.bin")
    save_bin(sorted_gsplat_positions, OutputDir + "gsplat_positions.bin")
    save_bin(sorted_anchor_scales, OutputDir + "anchor_scales.bin")
    save_bin(sorted_anchor_features, OutputDir + "anchor_features.bin")
    save_bin(sorted_anchor_levels, OutputDir + "anchor_levels.bin")
    save_bin(sorted_anchor_extra_levels, OutputDir + "anchor_extra_levels.bin")
    save_bin(anchor_info, OutputDir + "anchor_info.bin")

def count_elements(lst):
    count_dict = {}
    for item in lst:
        if int(item) in count_dict:
            count_dict[int(item)] += 1
        else:
            count_dict[int(item)] = 1
    for key, value in count_dict.items():
        print(f"{key}: {value}")

def save_bin(data, output_path):
    buffer = BytesIO()
    for x in data:
        y = np.array(x, dtype=np.float32)
        buffer.write(y.tobytes())
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())
    print(f"Saved {output_path}")

def convert_model(SourceDir, OutputDir, model_path):
    print(f"Model summary for {model_path}:")
    model = torch.jit.load( SourceDir + model_path + ".pt")
    print("Model information:")
    print(model)
    input_size = (1, 1, 1, 35)
    dummy_input = torch.randn(input_size)
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    output = model(dummy_input)
    print("\nModel input and output sizes:")
    print(f"Input size: {input_size}")
    print(f"Output size: {output.size()}")
    model.eval()
    torch.onnx.export(model, dummy_input, OutputDir + model_path + ".onnx", opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes)

if __name__ == "__main__":
    SourceDir = "./gs-data/"
    OutputDir = "./converted/"
    convert_model(SourceDir, OutputDir, "color_mlp")
    convert_model(SourceDir, OutputDir, "cov_mlp")
    convert_model(SourceDir, OutputDir, "opacity_mlp")
    convert_ply(SourceDir, OutputDir, "point_cloud.ply")