import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import re
from collections import defaultdict

features_dir = "saved_features_per_bbox"
filename_pattern = re.compile(r"frame_\d+_id_(\d+)_bbox_.*\.npy")

# Raggruppa le feature per track_id
features_by_id = defaultdict(list)

for fname in os.listdir(features_dir):
    match = filename_pattern.match(fname)
    if match:
        track_id = int(match.group(1))
        path = os.path.join(features_dir, fname)
        feat = np.load(path)
        feat = feat.flatten()  
        features_by_id[track_id].append(feat)

# PCA per ridurre a 2D
all_features = []
all_labels = []
for track_id, feats in features_by_id.items():
    for feat in feats:
        all_features.append(feat)
        all_labels.append(track_id)

all_features = np.array(all_features)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(all_features)

# Raggruppa le feature 2D per track_id
features_2d_by_id = defaultdict(list)
for point, label in zip(features_2d, all_labels):
    features_2d_by_id[label].append(point)

# Plot delle gaussiane
plt.figure(figsize=(10, 8))
colors = sns.color_palette("hls", len(features_2d_by_id))

for idx, (track_id, points) in enumerate(features_2d_by_id.items()):
    points = np.array(points)
    mean = points.mean(axis=0)
    cov = np.cov(points.T)

    rv = multivariate_normal(mean, cov)
    
    # Griglia per contour
    x, y = np.mgrid[features_2d[:,0].min():features_2d[:,0].max():.01,
                    features_2d[:,1].min():features_2d[:,1].max():.01]
    pos = np.dstack((x, y))
    z = rv.pdf(pos)

    plt.contour(x, y, z, levels=3, colors=[colors[idx]])
    plt.scatter(points[:, 0], points[:, 1], s=10, color=colors[idx], label=f'ID {track_id}')

plt.title("Distribuzioni Gaussiane delle ROI Feature (ridotte con PCA)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
