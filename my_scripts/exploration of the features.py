import torch

# Load the saved data
data = torch.load("masa_person_features_unfiltered.pth")
person_data = data['person_data']      # Raw records per track ID
person_stats = data['person_stats']      # Computed statistics per track ID

# Print overall stats for each track ID
for track_id, stats in person_stats.items():
    print(f"Track ID: {track_id}")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

# Optionally, inspect individual records
for track_id, records in person_data.items():
    print(f"\nRecords for Track ID {track_id}:")
    for record in records:
        # Here record['feature'] is a tensor. To convert to a list:
        feat_list = record['feature'].tolist()  
        print(f"  Frame {record['frame']}, BBox: {record['bbox']}, Feature: {feat_list[:5]} ...")
