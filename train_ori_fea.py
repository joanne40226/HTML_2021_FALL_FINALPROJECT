import argparse
import os
import random
import torch
import torch.optim as optim
import torch.utils.data
from pointnet.dataset_ori_fea import ShapeNetDataset  
from pointnet.model_ori_fea import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

print("PointNet - dataset : 3D points_Nx3  - feature_withoutLabel")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 修改過的部分：只進行特徵提取，無需label
dataset = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice])  # Remove classification-related code
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

# 無需進行標籤的測試
test_dataset = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

# 設定輸出文件夾
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

# 初始化 PointNet 模型，僅提取特徵
classifier = PointNetDenseCls(k=1, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))  # Ensure the model is loaded on CPU

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Ensure model is on CPU
classifier = classifier.cpu()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points = data  # 不需要label，僅關心point數據
        points = points.transpose(2, 1)
        points = points.cpu()  # 確保point數據在CPU上
        print(f"point shape: {points.shape}") 
        optimizer.zero_grad()
        classifier = classifier.train()
        features, trans, trans_feat = classifier(points)  # 只關心特徵提取，忽略pred
        print(f"features shape: {features.shape}") 
        if opt.feature_transform:
            loss = feature_transform_regularizer(trans_feat) * 0.001  # 使用正則化
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        loss.backward()

        optimizer.step()
        print('[%d: %d/%d] feature extraction loss: %f' % (epoch, i, num_batch, loss.item()))
        #print(f'feature: {features}')
        '''
        if i % 10 == 0:
            print('[%d: %d/%d] feature extraction loss: %f' % (epoch, i, num_batch, loss.item()))
        '''

    torch.save(classifier.state_dict(), '%s/feature_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

