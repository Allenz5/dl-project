import torch
import torch.nn as nn

from config import NUM_POINTS, NUM_CLASSES


class SortedMLP(nn.Module):
    def __init__(self, num_points=NUM_POINTS, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(num_points * 3, 512), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(512, 256),            nn.ReLU(), nn.Dropout(.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, N, _ = x.shape
        keys     = x[:, :, 0] * 1e8 + x[:, :, 1] * 1e4 + x[:, :, 2]   # (B, N)
        idx      = keys.argsort(dim=1).unsqueeze(-1).expand(B, N, 3)     # (B, N, 3)
        x_sorted = torch.gather(x, 1, idx)                               # (B, N, 3)
        return self.net(x_sorted.reshape(B, -1))


class PointNetVanilla(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU())
        self.enc = nn.Sequential(blk(3, 64), blk(64, 64), blk(64, 64), blk(64, 128), blk(128, 1024))
        self.cls = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.cls(self.enc(x.transpose(1, 2)).max(2)[0])


class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU())
        self.conv = nn.Sequential(blk(k, 64), blk(64, 128), blk(128, 1024))
        self.fc   = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.out = nn.Linear(256, k * k)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B = x.size(0)
        h = self.conv(x).max(2)[0]
        return (self.out(self.fc(h)).view(B, self.k, self.k)
                + torch.eye(self.k, device=x.device).unsqueeze(0))


class PointNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU())
        self.t3  = TNet(3)
        self.e1  = nn.Sequential(blk(3, 64),   blk(64, 64))
        self.t64 = TNet(64)
        self.e2  = nn.Sequential(blk(64, 64),  blk(64, 128), blk(128, 1024))
        self.cls = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(512,  256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def feat_reg(A):
        B, K, _ = A.shape
        I    = torch.eye(K, device=A.device).unsqueeze(0)
        diff = torch.bmm(A, A.transpose(1, 2)) - I
        return torch.mean(torch.norm(diff, dim=(1, 2)))

    def forward(self, x, return_feat=False):
        x    = x.transpose(1, 2)
        x    = torch.bmm(self.t3(x), x)
        x    = self.e1(x)
        A    = self.t64(x)
        x    = torch.bmm(A, x)
        x    = self.e2(x)
        feat = x.max(2)[0]
        logits = self.cls(feat)
        if return_feat:
            return logits, feat, A
        return logits
