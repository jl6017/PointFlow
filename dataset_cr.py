import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
import random

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

class PLYPointCloudDataset(Dataset):
    def __init__(self, root_dir,
                 tr_sample_size=10000,
                 te_sample_size=2048,
                 split='train',
                 normalize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_subsample=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3):
        self.root_dir = root_dir
        self.split = split
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".ply")])[:1000]
        if len(self.files) == 0:
            print(f"Directory missing or empty: {root_dir}")
        else:
            print(f"Loaded {len(self.files)} PLY files from {root_dir}")

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        self.random_subsample = random_subsample
        self.input_dim = input_dim

        # shuffle dataset order deterministically
        self.shuffle_idx = list(range(len(self.files)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.files = [self.files[i] for i in self.shuffle_idx]

        # load all shapes into memory (like ShapeNet loader)
        self.all_points = []
        self.cate_idx_lst = []
        self.all_cate_mids = []
        for idx, fname in enumerate(self.files):
            pcd = o3d.io.read_point_cloud(os.path.join(root_dir, fname))
            pts = np.asarray(pcd.points).astype(np.float32)  # (N, 3)
            if pts.shape[0] < 15000:
                # pad by repeating random points if fewer than 15k
                pad_idx = np.random.choice(pts.shape[0], 15000 - pts.shape[0], replace=True)
                pts = np.vstack([pts, pts[pad_idx]])
            pts = pts[:15000, :]
            self.all_points.append(pts[np.newaxis, ...])
            self.cate_idx_lst.append(0)  # dummy category index
            self.all_cate_mids.append(("ply", fname))

        self.all_points = np.concatenate(self.all_points)  # (B, 15000, 3)

        # normalization
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        # apply normalization
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        # split train/test
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        # visualization params (needed by train.py)
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        print(f"Total number of data: {len(self.train_points)}")
        print(f"Min number of points: (train){self.tr_sample_size} (test){self.te_sample_size}")

    def __len__(self):
        return len(self.train_points)

    def get_pc_stats(self, idx):
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s
        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]

        return {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m,
            'std': s,
            'cate_idx': cate_idx,
            'sid': sid,
            'mid': mid
        }


def get_datasets(args):
    tr_dataset = PLYPointCloudDataset(
        root_dir=args.data_dir,
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        split='train',
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        random_subsample=True
    )
    te_dataset = PLYPointCloudDataset(
        root_dir=args.data_dir,
        tr_sample_size=args.tr_max_sample_points,
        te_sample_size=args.te_max_sample_points,
        split='test',
        normalize_per_shape=args.normalize_per_shape,
        normalize_std_per_axis=args.normalize_std_per_axis,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std
    )
    return tr_dataset, te_dataset


if __name__ == "__main__":
    class DummyArgs:
        data_dir = "./../dataset_cr/pointcloud_15k/"
        tr_max_sample_points = 10000
        te_max_sample_points = 2048
        normalize_per_shape = False
        normalize_std_per_axis = False

    args = DummyArgs()
    tr_dataset, te_dataset = get_datasets(args)

    print("Train dataset size:", len(tr_dataset))
    print("Test dataset size:", len(te_dataset))

    sample = tr_dataset[0]
    print("Train points shape:", sample['train_points'].shape)
    print("Test points shape:", sample['test_points'].shape)
    print("Meta info:", sample['cate_idx'], sample['sid'], sample['mid'])
