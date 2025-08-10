import os
import re
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# --- Parser ---
def get_args():
    parser = argparse.ArgumentParser(description="ID association assistant")
    parser.add_argument("--feature_dir", type=str, default="saved_features_per_neck_2",
                        help="Directory con le feature .npy")
    parser.add_argument("--dim", type=int, default=32, help="Dimensione PCA")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Soglia distanza euclidea per candidati")
    parser.add_argument("--top_k", type=int, default=5, help="Numero massimo candidati da mostrare")
    parser.add_argument("--lock_after", type=int, default=5,
                        help="Frame consecutivi per fissare associazione")
    parser.add_argument("--output_csv", type=str, default="id_associations.csv",
                        help="File CSV per salvare le associazioni")
    return parser.parse_args()

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

def main():
    args = get_args()

    feature_files = sorted([f for f in os.listdir(args.feature_dir) if f.endswith('.npy')])
    features, metadata = [], []

    for fname in feature_files:
        frame_id, track_id, bbox = parse_filename(fname)
        if frame_id is None:
            continue
        feat = np.load(os.path.join(args.feature_dir, fname))
        features.append(feat.flatten())
        metadata.append((frame_id, track_id, bbox, fname))

    X = np.stack(features)
    X_pca = PCA(n_components=args.dim).fit_transform(X)

    all_models = {}
    id_aliases = {}
    recent_choices = defaultdict(lambda: deque(maxlen=args.lock_after))

    stats_total = stats_top1 = stats_top3 = stats_top5 = 0

    for i, (frame_id, track_id, bbox, fname) in enumerate(metadata):
        feat = X_pca[i]

        if track_id not in all_models and track_id not in id_aliases:
            if all_models:
                distances = []
                for other_id, model in all_models.items():
                    dist = model.euclidean_to(feat)
                    distances.append((other_id, dist))

                distances.sort(key=lambda x: x[1])
                top_candidates = [d for d in distances if d[1] < args.threshold][:args.top_k]

                if top_candidates:
                    # Lock automatico
                    recent_choices[track_id].append(top_candidates[0][0])
                    if (len(recent_choices[track_id]) == args.lock_after and 
                        len(set(recent_choices[track_id])) == 1):
                        chosen_id = top_candidates[0][0]
                        print(f"Auto-associazione fissata: {track_id} → {chosen_id}")
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
                            print(f"Associazione: {track_id} → {chosen_id}")
                            id_aliases[track_id] = chosen_id
                            all_models[chosen_id].update(feat)

                            # --- Statistiche ---
                            stats_total += 1
                            if choice_idx == 0:
                                stats_top1 += 1
                            if choice_idx < 3:
                                stats_top3 += 1
                            if choice_idx < 5:
                                stats_top5 += 1
                            continue

            all_models[track_id] = AppearanceModel(dim=args.dim)

        target_id = id_aliases.get(track_id, track_id)
        all_models[target_id].update(feat)

    pd.DataFrame([
        {'new_id': k, 'associated_to': v} for k, v in id_aliases.items()
    ]).to_csv(args.output_csv, index=False)

    print("\nAssociazioni salvate in", args.output_csv)
    if stats_total > 0:
        print(f"\n Statistiche sulle scelte:")
        print(f"  Top-1: {stats_top1 / stats_total * 100:.2f}%")
        print(f"  Top-3: {stats_top3 / stats_total * 100:.2f}%")
        print(f"  Top-5: {stats_top5 / stats_total * 100:.2f}%")

if __name__ == "__main__":
    main()
