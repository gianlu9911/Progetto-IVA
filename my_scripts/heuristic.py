import os
import re
import argparse
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

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

def main(args):
    pause_frames = int(args.pause_sec * args.fps)
    last_choice_frame = {}

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

    frame_to_ids = defaultdict(set)
    for (frame_id, track_id, bbox, fname) in metadata:
        frame_to_ids[frame_id].add(track_id)

    all_models = {}
    id_aliases = {}
    alias_frame = {}
    recent_choices = defaultdict(lambda: deque(maxlen=args.lock_after))

    skip_ids = set()

    total_choices = 0
    top1_count = 0
    top2_count = 0
    top3_count = 0

    for i, (frame_id, track_id, bbox, fname) in enumerate(metadata):
        if track_id in skip_ids:
            continue

        feat = X_pca[i]
        target_id = id_aliases.get(track_id, track_id)

        if target_id in skip_ids:
            continue

        # Pause period
        if track_id in last_choice_frame:
            if frame_id - last_choice_frame[track_id] < pause_frames:
                if target_id not in skip_ids:
                    all_models[target_id].update(feat)
                continue

        frame_ids_present = frame_to_ids[frame_id]
        prev_frame_ids = frame_to_ids.get(frame_id - 1, set())

        frame_ids_present = {tid for tid in frame_ids_present if tid not in skip_ids}
        prev_frame_ids = {tid for tid in prev_frame_ids if tid not in skip_ids}

        # Aliases already present in this frame (skip candidates with these)
        frame_aliases_present = {id_aliases.get(tid, tid) for tid in frame_ids_present}

        # --- Conflict resolution ---
        if track_id in id_aliases and track_id not in skip_ids:
            original_id = id_aliases[track_id]
            if original_id in skip_ids:
                continue

            related_ids = {oid for oid, aid in id_aliases.items() if aid == original_id and oid not in skip_ids}
            related_ids.add(original_id)

            if any(rid in frame_ids_present for rid in related_ids if rid != track_id):
                print(f"\nConflitto: ID {track_id} (alias {original_id}) compare insieme ad {related_ids}")
                distances = []
                for other_id, model in all_models.items():
                    if other_id in skip_ids:
                        continue
                    # Skip aliases already present in frame
                    #if other_id in frame_aliases_present:
                        #continue
                    dist = model.euclidean_to(feat)
                    distances.append((other_id, dist))
                distances.sort(key=lambda x: x[1])

                # Rimuovo duplicati ID mantenendo ordine crescente distanza
                unique_candidates = []
                seen_ids = set()
                for oid, dist in distances:
                    if oid not in seen_ids:
                        unique_candidates.append((oid, dist))
                        seen_ids.add(oid)

                top_candidates = [d for d in unique_candidates if d[1] < args.euclidean_threshold][:args.top_k]

                print(f"[Frame {frame_id}] Possibili nuove associazioni per {track_id}:")
                for idx, (oid, dist) in enumerate(top_candidates, start=1):
                    print(f"  {idx}) ID {oid} - dist {dist:.2f}")
                print(f"  s) Salta (skip) questa ID completamente")

                choice = input(f"Seleziona 1-{len(top_candidates)}, 's' per skip, o ENTER per nuova ID: ").strip()
                if choice == "":
                    all_models[track_id] = AppearanceModel(dim=args.dim)
                    all_models[track_id].update(feat)
                    last_choice_frame[track_id] = frame_id
                    continue
                elif choice == "s":
                    print(f"ID {track_id} aggiunta a skip_ids, non verrà più considerata.")
                    skip_ids.add(track_id)
                    continue
                elif choice.isdigit():
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(top_candidates):
                        chosen_id = top_candidates[choice_idx][0]
                        print(f"Riassegnato: {track_id} → {chosen_id}")
                        id_aliases[track_id] = chosen_id
                        alias_frame[track_id] = frame_id
                        all_models[chosen_id].update(feat)
                        last_choice_frame[track_id] = frame_id
                        continue

        # --- New track_id ---
        if track_id not in all_models and track_id not in id_aliases and track_id not in skip_ids:
            if all_models:
                distances = []
                for other_id, model in all_models.items():
                    # Calcoliamo comunque per tutti i modelli, anche se sarebbero stati skippati
                    dist = model.euclidean_to(feat)
                    distances.append((other_id, dist))

                # Migliore globale (anche sopra soglia)
                global_best = min(distances, key=lambda x: x[1]) if distances else None

                # Filtriamo solo per i candidati stampabili
                filtered_candidates = [
                    (oid, dist) for oid, dist in distances
                    if oid not in skip_ids and oid not in prev_frame_ids and oid not in frame_aliases_present
                ]

                filtered_candidates.sort(key=lambda x: x[1])
                unique_candidates = []
                seen_ids = set()
                for oid, dist in filtered_candidates:
                    if oid not in seen_ids:
                        unique_candidates.append((oid, dist))
                        seen_ids.add(oid)

                top_candidates = [d for d in unique_candidates if d[1] < args.euclidean_threshold][:args.top_k]


                if top_candidates:
                    # Trova il migliore candidato globale (anche sopra soglia)
                    global_best = min(distances, key=lambda x: x[1]) if distances else None

                    recent_choices[track_id].append(top_candidates[0][0])
                    if len(recent_choices[track_id]) == args.lock_after and len(set(recent_choices[track_id])) == 1:
                        chosen_id = top_candidates[0][0]
                        print(f"Auto-associazione fissata: {track_id} → {chosen_id}")
                        id_aliases[track_id] = chosen_id
                        alias_frame[track_id] = frame_id
                        all_models[chosen_id].update(feat)
                        last_choice_frame[track_id] = frame_id
                        continue

                    print(f"\n[Frame {frame_id}] Nuova ID {track_id} - possibili associazioni:")
                    for idx, (oid, dist) in enumerate(top_candidates, start=1):
                        print(f"  {idx}) ID {oid} - dist {dist:.2f}")

                    # Aggiungiamo opzione x
                    if global_best:
                        print(f"  x) ID {global_best[0]} - dist {global_best[1]:.2f}  (migliore tra tutti)")

                    print(f"  s) Salta (skip) questa ID completamente")


                    choice = input(f"Seleziona 1-{len(top_candidates)}, 'x' per migliore globale, 's' per skip, o ENTER per nuova ID: ").strip()
                    if choice == "":
                        all_models[track_id] = AppearanceModel(dim=args.dim)
                        all_models[track_id].update(feat)
                        last_choice_frame[track_id] = frame_id
                        total_choices += 1
                        continue
                    elif choice == "s":
                        print(f"ID {track_id} aggiunta a skip_ids, non verrà più considerata.")
                        skip_ids.add(track_id)
                        continue
                    elif choice == "x" and global_best:
                        chosen_id = global_best[0]
                        print(f"Associazione (forzata): {track_id} → {chosen_id}")
                        id_aliases[track_id] = chosen_id
                        alias_frame[track_id] = frame_id
                        all_models[chosen_id].update(feat)
                        last_choice_frame[track_id] = frame_id
                        total_choices += 1
                        continue
                    elif choice.isdigit():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(top_candidates):
                            chosen_id = top_candidates[choice_idx][0]
                            print(f"Associazione: {track_id} → {chosen_id}")
                            id_aliases[track_id] = chosen_id
                            alias_frame[track_id] = frame_id
                            all_models[chosen_id].update(feat)
                            last_choice_frame[track_id] = frame_id
                            total_choices += 1
                            if choice_idx == 0:
                                top1_count += 1
                            if choice_idx < 2:
                                top2_count += 1
                            if choice_idx <= 2:
                                top3_count += 1
                            continue

            all_models[track_id] = AppearanceModel(dim=args.dim)

        target_id = id_aliases.get(track_id, track_id)
        if target_id not in skip_ids:
            all_models[target_id].update(feat)

    filtered_aliases = {k: v for k, v in id_aliases.items() if k not in skip_ids and v not in skip_ids}

    pd.DataFrame([
        {'new_id': k, 'associated_to': v, 'frame': alias_frame.get(k, None)}
        for k, v in filtered_aliases.items()
    ]).to_csv("id_associations.csv", index=False)

    if total_choices == 0:
        total_choices = 1
        top1_count = 1
        top2_count = 1
        top3_count = 1

    print("\nStatistiche:")
    print(f"  Top1: {top1_count}/{total_choices} ({(top1_count/total_choices*100):.1f}%)")
    print(f"  Top2: {top2_count}/{total_choices} ({(top2_count/total_choices*100):.1f}%)")
    print(f"  Top3: {top3_count}/{total_choices} ({(top3_count/total_choices*100):.1f}%)")
    print("Associazioni salvate in id_associations.csv")
    print(f"ID skip: {sorted(skip_ids)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID association tool with conflict resolution and skip feature")
    parser.add_argument("--feature_dir", type=str, default="saved_features_per_neck_2", help="Directory con le feature .npy")
    parser.add_argument("--dim", type=int, default=16, help="Dimensione dopo PCA")
    parser.add_argument("--euclidean_threshold", type=float, default=80.0, help="Soglia distanza euclidea")
    parser.add_argument("--top_k", type=int, default=5, help="Numero massimo di candidati")
    parser.add_argument("--lock_after", type=int, default=5, help="Blocca auto-associazione dopo N scelte uguali consecutive")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS del video")
    parser.add_argument("--pause_sec", type=float, default=5.0, help="Durata pausa in secondi prima di rivalutare una ID")
    args = parser.parse_args()

    main(args)

