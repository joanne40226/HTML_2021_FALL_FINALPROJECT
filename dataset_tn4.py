import torch
import torch.utils.data as data
import os
import numpy as np
import json
from tqdm import tqdm


class TimeSeriesPointDataset(data.Dataset):
    def __init__(self, root, npoints=2500, split='train', data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation

        # Load train/test split
        splitfile = os.path.join(self.root, 'train_test_split', f'shuffled_{split}_file_list.json')
        with open(splitfile, 'r') as f:
            self.sequence_paths = json.load(f)

    def __getitem__(self, index):
        sequence_path = self.sequence_paths[index]
        seq_dir = os.path.join(self.root, sequence_path)
        time_steps = sorted([f for f in os.listdir(seq_dir) if f.endswith('.pts')])

        sequence = []
        for t_file in time_steps:
            point_set = np.loadtxt(os.path.join(seq_dir, t_file)).astype(np.float32)

            # Center and scale
            point_set = point_set[:self.npoints]
            point_set -= np.mean(point_set, axis=0)
            point_set /= np.max(np.linalg.norm(point_set, axis=1))

            if self.data_augmentation:
                theta = np.random.uniform(0, np.pi * 2)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
                point_set += np.random.normal(0, 0.02, size=point_set.shape)

            sequence.append(point_set)

        # Convert sequence to numpy array first, then to tensor
        sequence = np.array(sequence)  # Convert list of point_sets into a numpy array
        sequence = torch.tensor(sequence)  # Now convert the numpy array into a tensor

        return sequence

    def __len__(self):
        return len(self.sequence_paths)


if __name__ == '__main__':
    dataset = TimeSeriesPointDataset(root='formatted_dataset', split='train')
    print(len(dataset))
    sequence = dataset[0]
    print(sequence.shape)

