import argparse
import os
import random
import torch
import torch.optim as optim
from pointnet.dataset_tn4 import TimeSeriesPointDataset
from pointnet.model_tn4 import PointNetDenseCls, feature_transform_regularizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()

print(opt)

# Set random seed
random.seed(42)
torch.manual_seed(42)

# Load dataset
train_dataset = TimeSeriesPointDataset(root=opt.dataset, split='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)

classifier = PointNetDenseCls(k=1, feature_transform=opt.feature_transform).cpu()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))

for epoch in range(opt.nepoch):
    for i, data in enumerate(train_loader):
        data = data.transpose(1,2).cpu()  # t x N x 4 -> B x t x 4 x N
        optimizer.zero_grad()
        features, trans, trans_feat = classifier(data)
        loss = feature_transform_regularizer(trans_feat) if opt.feature_transform else torch.tensor(0.0)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Iter {i}, Loss: {loss.item()}')

