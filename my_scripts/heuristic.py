from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import os, re
from sklearn.decomposition import PCA

# --- Params ---
feature_dir = 'saved_features_per_neck'
dim = 32
EUCLIDEAN_THRESHOLD = 80.0
TOP_K = 5
LOCK_AFTER = 5  # Numero di volte consecutive per fissare l'associazione

# --- Utility ---
def parse_filename(fname):
    match = re.search(r"frame_(\d+)_id_(\d+)_bbox_(\d+)_(\d+)_(\d+)_(\d+)", fname)
    if match:
        frame_id = int(match.group(1))
        track_id = int(match.group(2))
        bbox = tuple(map(int, match.group(3, 4, 5, 6)))
        return frame_id, track_id, bbox
    return None, None, None

class AppearanceModel:
    def __init__(self, dim, alpha=0.1):
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self.count = 0
        self.alpha = alpha

    def update(self, x):
        x = x.reshape(-1)
        if self.count == 0:
            self.mean = x
            self.cov = np.eye(len(x))
        else:
            delta = x - self.mean
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
            self.cov = (1 - self.alpha) * self.cov + self.alpha * np.outer(delta, delta)
        self.count += 1

    def euclidean_to(self, x):
        return euclidean(x, self.mean)

# --- Load features ---
feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npy')])
features, metadata = [], []

for fname in feature_files:
    frame_id, track_id, bbox = parse_filename(fname)
    if frame_id is None:
        continue
    feat = np.load(os.path.join(feature_dir, fname))
    features.append(feat.flatten())
    metadata.append((frame_id, track_id, bbox, fname))

X = np.stack(features)
X_pca = PCA(n_components=dim).fit_transform(X)

# --- Tracking ---
all_models = {}  # track_id -> AppearanceModel
id_aliases = {}  # new_id -> old_id association
recent_choices = defaultdict(lambda: deque(maxlen=LOCK_AFTER))  # track_id -> ultimi candidati scelti

for i, (frame_id, track_id, bbox, fname) in enumerate(metadata):
    feat = X_pca[i]

    if track_id not in all_models and track_id not in id_aliases:
        if all_models:
            distances = []
            for other_id, model in all_models.items():
                dist = model.euclidean_to(feat)
                distances.append((other_id, dist))

            distances.sort(key=lambda x: x[1])
            top_candidates = [d for d in distances if d[1] < EUCLIDEAN_THRESHOLD][:TOP_K]

            if top_candidates:
                # Controlla se negli ultimi N frame ha scelto sempre lo stesso
                recent_choices[track_id].append(top_candidates[0][0])
                if len(recent_choices[track_id]) == LOCK_AFTER and len(set(recent_choices[track_id])) == 1:
                    chosen_id = top_candidates[0][0]
                    print(f"🔒 Auto-associazione fissata: {track_id} → {chosen_id}")
                    id_aliases[track_id] = chosen_id
                    all_models[chosen_id].update(feat)
                    continue

                print(f"\n[Frame {frame_id}] Nuova ID {track_id} - possibili associazioni:")
                for idx, (oid, dist) in enumerate(top_candidates, start=1):
                    print(f"  {idx}) ID {oid} - dist {dist:.2f}")

                choice = input(f"Seleziona 1-{len(top_candidates)} per associare, o ENTER per ignorare: ").strip()
                if choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(top_candidates):
                        chosen_id = top_candidates[choice_idx][0]
                        print(f"✅ Associazione: {track_id} → {chosen_id}")
                        id_aliases[track_id] = chosen_id
                        all_models[chosen_id].update(feat)
                        continue

        all_models[track_id] = AppearanceModel(dim=dim)

    target_id = id_aliases.get(track_id, track_id)
    all_models[target_id].update(feat)

# --- Save alias ---
pd.DataFrame([
    {'new_id': k, 'associated_to': v} for k, v in id_aliases.items()
]).to_csv("id_associations.csv", index=False)

print("\nAssociazioni salvate in id_associations.csv")
