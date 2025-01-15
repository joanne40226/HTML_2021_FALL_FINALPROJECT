import torch
import torch.nn as nn
import torch.nn.functional as F

print("PointNet - dataset : 3D points_Nx3  - feature_withoutLabel")

class PointNetDenseCls(nn.Module):
    def __init__(self, k=40, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform

        # 定義PointNet網絡結構
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # 定義全連接層
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        # 定義特徵變換矩陣
        self.fstn = FeatureTransformNet()

    def forward(self, x):
        # 輸入的x的形狀為(B, C, N)，其中C是通道數（通常為3），N是點的數量
        batchsize = x.size()[0]

        # PointNet網絡的前向傳播
        x = F.relu(self.conv1(x))
        # print("After Conv1:", x.shape)
        x = F.relu(self.conv2(x))
        # print("After Conv2:", x.shape)
        x = F.relu(self.conv3(x))
        # print("After Conv3:", x.shape)


        # 全局最大池化，聚合所有點的特徵
        x = torch.max(x, 2, keepdim=False)[0]

        # 全連接層
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred = self.fc3(x)

        # 特徵變換正則化
        if self.feature_transform:
            trans = self.fstn(x)
            return x, trans, trans
        else:
            return x, None, None

# 特徵變換網絡（Feature Transform Network）
class FeatureTransformNet(nn.Module):
    def __init__(self):
        super(FeatureTransformNet, self).__init__()

        # 用於生成特徵變換矩陣的網絡
        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 9)  # 用於生成 3x3 的特徵變換矩陣

    def forward(self, x):
        # 特徵變換網絡的前向傳播
        batchsize = x.size()[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 全連接層
        x = x.max(2, keepdim=False)[0]  # 最大池化
        x = F.relu(self.fc1(x))
        trans = self.fc2(x)

        # 重新設置為 3x3 矩陣
        trans = trans.view(-1, 3, 3)

        # 返回的轉換矩陣
        return trans

# 正則化項計算
def feature_transform_regularizer(trans):
    # 計算正則化項，用於限制學習到的變換矩陣
    I = torch.eye(3).cuda()  # 3x3 單位矩陣
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, p='fro'))
    loss = torch.tensor(loss, dtype=torch.float32)

    return loss

