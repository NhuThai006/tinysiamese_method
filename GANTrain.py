"""
Train ACGAN on 4 fingerprint classes (Arch, LeftLoop, RightLoop, Whorl)
and save Discriminator weights to model/GANFVC.h5 for feature extraction.

Run (from this folder):
    python GANTrain.py
"""
import os
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from gan_model import (
    NUM_CLASSES,
    IMAGE_SIZE,
    LATENT_DIM,
    build_gan,
)


INPUT_PATH = "Database/Train"
OUTPUT_PATH = "model/GANFVC.h5"
BATCH_SIZE = 16
EPOCHS = 50
LR = 2e-4
BETA1 = 0.5
REPEATS_PER_IMAGE = 3


def is_readable_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


class RepeatedFolder(Dataset):
    """ImageFolder wrapper that skips corrupt files and repeats with augmentation."""

    def __init__(self, folder, transform, repeats):
        self.base = datasets.ImageFolder(folder)
        skipped = []
        valid = []
        for path, label in self.base.samples:
            if is_readable_image(path):
                valid.append((path, label))
            else:
                skipped.append(path)
        if skipped:
            print("Skipped unreadable files (%d):" % len(skipped))
            for path in skipped:
                print(" -", path)
        self.samples = valid
        self.transform = transform
        self.repeats = max(1, repeats)
        if len(self.base.classes) != NUM_CLASSES:
            raise RuntimeError(
                "Can train GANFVC only with exactly 4 class folders under "
                "Database/Train (Arch, LeftLoop, RightLoop, Whorl). Found: "
                + str(self.base.classes)
            )
        if not self.samples:
            raise RuntimeError("No readable images found in Database/Train.")

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        real_index = index % len(self.samples)
        path, label = self.samples[real_index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def main():
    if not os.path.isdir(INPUT_PATH):
        raise FileNotFoundError(
            "Missing folder %s. Put fingerprint images into "
            "Database/Train/<class>/ first." % INPUT_PATH
        )

    os.makedirs("model", exist_ok=True)

    # Tanh output of Generator expects images in [-1, 1]
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = RepeatedFolder(INPUT_PATH, train_tf, REPEATS_PER_IMAGE)
    print("Classes:", dataset.base.classes, flush=True)
    print("Readable images:", len(dataset.samples), "| Augmented/epoch:", len(dataset), flush=True)
    if len(dataset.samples) < NUM_CLASSES:
        raise RuntimeError(
            "Need at least 1 image in each of the 4 class folders before training."
        )

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=True if len(dataset) >= BATCH_SIZE else False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    G, D = build_gan(device)
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    opt_G = Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    best_aux_acc = -1.0
    best_D_state = None

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}...", flush=True)
        G.train()
        D.train()

        sum_d_loss = 0.0
        sum_g_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        for real_imgs, labels in loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # ----- Train Discriminator -----
            opt_D.zero_grad()

            real_pred, real_aux = D(real_imgs)
            d_real_loss = (
                adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)
            ) / 2

            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            fake_imgs = G(z, gen_labels)

            fake_pred, fake_aux = D(fake_imgs.detach())
            d_fake_loss = (
                adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)
            ) / 2

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # ----- Train Generator -----
            opt_G.zero_grad()

            fake_pred, fake_aux = D(fake_imgs)
            g_loss = (
                adversarial_loss(fake_pred, valid) + auxiliary_loss(fake_aux, gen_labels)
            ) / 2
            g_loss.backward()
            opt_G.step()

            # Aux accuracy on real images (feature quality signal)
            preds = real_aux.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            sum_d_loss += d_loss.item()
            sum_g_loss += g_loss.item()
            n_batches += 1

        avg_d = sum_d_loss / max(n_batches, 1)
        avg_g = sum_g_loss / max(n_batches, 1)
        aux_acc = 100.0 * correct / max(total, 1)
        print(
            "D_loss: %.4f  G_loss: %.4f  aux_acc(real): %.2f%%"
            % (avg_d, avg_g, aux_acc),
            flush=True,
        )

        if aux_acc >= best_aux_acc:
            best_aux_acc = aux_acc
            best_D_state = copy.deepcopy(D.state_dict())

    torch.save(best_D_state if best_D_state is not None else D.state_dict(), OUTPUT_PATH)
    # Also keep generator for optional inspection / future use
    torch.save(G.state_dict(), "model/GAN_Generator.h5")
    print("Saved", OUTPUT_PATH, "best_aux_acc=%.2f%%" % best_aux_acc, flush=True)
    print("Saved model/GAN_Generator.h5", flush=True)
    print(
        "Note: with only a few images this file exists so TinySiamese can run, "
        "but quality improves when you add more FVC images per class.",
        flush=True,
    )


if __name__ == "__main__":
    main()
