from math import copysign
from peopleTracker import deserialize_track_data, serialize_track_data
import argparse



def how_many_tracks(tracks):
    print(f"There are {len(tracks)} track(s)")



def remove_hops(tracks):
    for track_id, (track_color, track_data) in tracks.items():
        if len(track_data) < 2:
            continue
        frame_idxs = list(track_data.keys())
        bbs = [bb for point, x_couple, y_couple, bb in track_data.values()]

        bb_heights = [abs((bb[1] - bb[3])) for bb in bbs]
        bb_avg_height = sum(bb_heights) / len(bb_heights)
        bb_std_dev_height = (sum([(bb_height - bb_avg_height) ** 2 for bb_height in bb_heights]) / len(bb_heights)) ** 0.5
        std_dev_scale_factor = 0.5
        found_height_idx = 0
        found_height = 0
        for idx, (bb, bb_height) in enumerate(zip(bbs, bb_heights)):
            if abs(bb_height - bb_avg_height) <= std_dev_scale_factor * bb_std_dev_height:
                found_height_idx = idx
                found_height = bb_height
                break
        start_idx = 0
        start_height_diff_ratio_thres = 0.15
        for idx, (prev_bb_height, next_bb_height) in enumerate(zip(bb_heights, bb_heights[1:])):
            if idx == found_height_idx:
                break
            bb_height_diff = next_bb_height - prev_bb_height
            max_bb_height = max(prev_bb_height, next_bb_height)
            bb_height_diff_ratio = abs(bb_height_diff) / max_bb_height
            if bb_height_diff_ratio >= start_height_diff_ratio_thres:
                start_idx = idx
                break
        bbs[start_idx : found_height_idx] = [[bb[0], bb[1], bb[2], bb[1] + found_height] for bb in bbs[start_idx : found_height_idx]]
        del bbs[0 : start_idx]
        for frame_idx in frame_idxs[0 : start_idx]:
            del tracks[track_id][1][frame_idx]
        del frame_idxs[0 : start_idx]
        
        new_bbs = [bbs[0]] 
        height_diff_ratio_thres = 0.0075
        for next_bb in bbs[1:]:
            prev_bb = new_bbs[-1]
            prev_bb_height = abs(prev_bb[1] - prev_bb[3])
            next_bb_height = abs(next_bb[1] - next_bb[3])
            bb_height_diff = next_bb_height - prev_bb_height
            max_bb_height = max(prev_bb_height, next_bb_height)
            bb_height_diff_ratio = abs(bb_height_diff) / max_bb_height
            if bb_height_diff_ratio > height_diff_ratio_thres:
                new_bb = next_bb
                new_bb[3] = new_bb[1] + prev_bb_height + copysign(height_diff_ratio_thres * max_bb_height, bb_height_diff)
                new_bbs.append(new_bb)
            else:
                new_bbs.append(next_bb)
        
        points = [((bb[0] + bb[2]) / 2, bb[3]) for bb in new_bbs]
        x_couples = [(prev_point[0], next_point[0]) for prev_point, next_point in  zip(points, points[1:])]
        x_couples.insert(0, (0.0, 0.0))
        y_couples = [(prev_point[1], next_point[1]) for prev_point, next_point in  zip(points, points[1:])]
        y_couples.insert(0, (0.0, 0.0))
        
        for (f, p, x_c, y_c, bb) in zip(frame_idxs, points, x_couples, y_couples, new_bbs):    
            tracks[track_id][1][f] = (p, x_c, y_c, bb)



def pair_diff(a, b):
    return (a[0] - b[0], a[1] - b[1])

def pair_magn(pair):
    return (pair[0] ** 2 + pair[1] ** 2) ** 0.5

def cos_sim(a, b):
    scalar_prod = a[0] * b[0] + a[1] * b[1]
    a_magn = pair_magn(a)
    b_magn = pair_magn(b)
    if a_magn * b_magn != 0:
        return scalar_prod / (a_magn * b_magn)
    else:
        return 0

def get_first_frame_idx_pckd(track_pckd):
    return int(next(iter(track_pckd[1].keys())))

def get_first_frame_idx(track):
    return int(next(iter(track.keys())))

def get_last_frame_idx(track):
    return int(next(iter(reversed(track.keys()))))

def does_it_end_near_start_of(targ_track, ref_track, frame_interval):
    end_frame_idx = get_first_frame_idx(ref_track)
    last_frame_idx = get_last_frame_idx(targ_track)
    if abs(end_frame_idx - last_frame_idx) <= frame_interval:
        return True
    else:
        return False

def are_last_bbs_and_first_bb_spatially_near(targ_track, ref_track, frame_interval):
    last_frame_idx, (p, x_c, y_c, (f_x1, upper_bound, f_x2, lower_bound)) = next(iter(ref_track.items()))
    f_c_x = (f_x1 + f_x2) / 2
    f_x_len = abs(f_x1 - f_x2)
    left_bound = f_c_x - f_x_len
    right_bound = f_c_x + f_x_len

    for frame_idx, (p, x_c, y_c, (x1, y1, x2, y2)) in reversed(targ_track.items()):
        if abs(int(frame_idx) - int(last_frame_idx)) > frame_interval:
            break
        if (x1 <= right_bound and x2 >= left_bound) and (y1 <= lower_bound and y2 >= upper_bound):
            return True
    return False

def score_candidate(ref_track_data, candidate_track, frame_interval):
    ref_first_point = next(iter(ref_track_data.values()))[0]
    ref_first_idx = get_first_frame_idx(ref_track_data)
    ref_next_point = ref_first_point
    for frame_idx, (point, x_c, y_c, bb) in ref_track_data.items():
        if int(frame_idx) - ref_first_idx <= frame_interval:
            ref_next_point = point
    
    cand_last_point = next(iter(reversed(candidate_track[1].values())))[0]
    cand_last_frame_idx = get_last_frame_idx(candidate_track[1])
    cand_prev_point = cand_last_point
    for frame_idx, (point, x_c, y_C, bb) in reversed(candidate_track[1].items()):
        if cand_last_frame_idx - int(frame_idx) <= frame_interval:
            cand_prev_point = point 

    ref_vec = pair_diff(ref_next_point, ref_first_point)
    cand_vec = pair_diff(cand_last_point, cand_prev_point)
    return cos_sim(ref_vec, cand_vec)

def fuse_track_into_cand(track_data, candidate):
    first_frame_idx, (next_p, next_x_c, next_y_c, next_bb) = next(iter(track_data.items()))
    prev_p, prev_x_c, prev_y_c, prev_bb = next(iter(reversed(candidate[1].values())))
    next_x_c = (next_p[0], prev_p[0])
    next_y_c = (next_p[1], prev_p[1])
    candidate[1][first_frame_idx] = (next_p, next_x_c, next_y_c, next_bb)
    for (frame_idx, (p, x_c, y_c, bb)) in list(track_data.items())[1:]:
        candidate[1][frame_idx] = (p, x_c, y_c, bb)

def fuse_tracks(tracks):
    m_tracks = [(track_id, track_data) for track_id, (track_color, track_data) in tracks.items()]
    m_tracks.sort(key=get_first_frame_idx_pckd, reverse=True)
    frame_interval = 30
    for (idx, (track_id, track_data)) in enumerate(m_tracks[:-1]):
        candidates = [track for track in m_tracks[idx + 1:]
                      if does_it_end_near_start_of(track[1], track_data, frame_interval) and
                        are_last_bbs_and_first_bb_spatially_near(track[1], track_data, frame_interval)]
        if not candidates:
            continue
        candidates.sort(key= lambda candidate : score_candidate(track_data, candidate, frame_interval),
                        reverse=True)
        best_candidate = candidates[0]
        fuse_track_into_cand(track_data, best_candidate)
        del tracks[track_id]



def parse_args():
    parser = argparse.ArgumentParser(prog="Tracks filters")

    parser.add_argument("--in_tracks")
    parser.add_argument("--out_tracks")

    parser.add_argument("--how_many", action="store_true")

    parser.add_argument("--f_rem_hops", action="store_true")
    parser.add_argument("--f_fuse_tracks", action="store_true")
    
    return parser.parse_args()

def check_args(args):
    if not args.in_tracks:
        print("You have to specify on which tracks you want to work (--in_tracks)")
        exit()

    if (args.f_rem_hops or args.f_fuse_tracks) and not args.out_tracks:
        print("You have to specify where to save the filtered tracks (--out_tracks)")
        exit()

def main():
    args = parse_args()
    check_args(args)
    tracks = deserialize_track_data(args.in_tracks)

    if args.f_rem_hops:
        remove_hops(tracks)
        serialize_track_data(args.out_tracks, tracks)
    elif args.f_fuse_tracks:
        fuse_tracks(tracks)
        serialize_track_data(args.out_tracks, tracks)
    elif args.how_many:
        how_many_tracks(tracks)

    print("Done")

if __name__ == "__main__":
    main()