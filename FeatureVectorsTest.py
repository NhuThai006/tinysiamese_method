# Extract 4096-d feature vectors from Test images using ACGAN Discriminator
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from gan_model import IMAGE_SIZE, Discriminator


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


input_path = "Database/"

# Match GAN training normalization (Tanh space: [-1, 1])
data_transforms = {
    "Test": transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
}

image_datasets = {
    "Test": ImageFolderWithPaths(input_path + "Test", data_transforms["Test"])
}

dataloaders = {
    "Test": DataLoader(
        image_datasets["Test"],
        batch_size=1,
        shuffle=True,
        num_workers=0,
    ),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminator().to(device)
model.load_state_dict(torch.load("model/GANFVC.h5", map_location=device))
model.eval()
print(model)

ftest = open("csv_test.csv", "w", newline="")
ftestLabels = open("csv_testLabels.csv", "w", newline="")
ftestPaths = open("csv_testPaths.csv", "w", newline="")
writer_feat = csv.writer(ftest)
writer_labels = csv.writer(ftestLabels)
writer_paths = csv.writer(ftestPaths)

with torch.no_grad():
    for images, labels, path in dataloaders["Test"]:
        images = images.to(device)
        feat = model.extract_features(images)
        feat = feat.flatten()
        writer_feat.writerow(feat.cpu().numpy())
        writer_labels.writerow(labels.numpy())
        writer_paths.writerow(path)

ftest.close()
ftestLabels.close()
ftestPaths.close()
print("Wrote csv_test.csv, csv_testLabels.csv, csv_testPaths.csv")
