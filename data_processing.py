import os
import open3d as o3d
from tqdm import tqdm

def preprocess_ply(input_dir, output_dir, n_samples=15000, max_files=1000):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".ply")])
    if max_files > 0:
        files = files[:max_files]

    print(f"Processing {len(files)} .ply files...")

    for fname in tqdm(files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # load
        pcd = o3d.io.read_point_cloud(in_path)

        # Open3D FPS downsample
        if len(pcd.points) > n_samples:
            pcd_ds = pcd.farthest_point_down_sample(n_samples)
        else:
            pcd_ds = pcd  # if already small, keep as is

        # save
        o3d.io.write_point_cloud(out_path, pcd_ds)

    print(f"Saved {len(files)} downsampled PLY files to {output_dir}")


if __name__ == "__main__":
    input_dir = "./../dataset_cr/pointcloud/"
    output_dir = "./../dataset_cr/pointcloud_15k/"
    preprocess_ply(input_dir, output_dir, n_samples=15000, max_files=1000)
