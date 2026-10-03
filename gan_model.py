"""
ACGAN (Auxiliary Classifier GAN) for fingerprint feature extraction.

Discriminator produces 4096-d features so TinySiameseTrain.py stays unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 4
FEATURE_DIM = 4096
IMAGE_SIZE = 64
LATENT_DIM = 100
CHANNELS = 3


def _conv_out_size():
    # 64 -> 32 -> 16 -> 8 -> 4, channels 512
    return 512 * 4 * 4


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, channels=CHANNELS):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.deconv(x)


class Discriminator(nn.Module):
    """
    Conv backbone + 4096-d feature layer, then:
      - adversarial head (real/fake)
      - auxiliary class head (4 fingerprint types)
    """

    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM, channels=CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        flat = _conv_out_size()
        self.fc_feat = nn.Linear(flat, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc_adv = nn.Linear(feature_dim, 1)
        self.fc_aux = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        feat = F.relu(self.fc_feat(h))
        return self.dropout(feat)

    def forward(self, x, return_features=False):
        feat = self.extract_features(x)
        validity = torch.sigmoid(self.fc_adv(feat))
        class_logits = self.fc_aux(feat)
        if return_features:
            return validity, class_logits, feat
        return validity, class_logits


def build_gan(device=None):
    G = Generator()
    D = Discriminator()
    if device is not None:
        G = G.to(device)
        D = D.to(device)
    return G, D
