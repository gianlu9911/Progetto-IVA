import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings("ignore")

import argparse
import cv2
import gc
import math
import json
#
from torchvision.ops import roi_align
#
import torch
from collections import defaultdict
from torch.multiprocessing import Pool

import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmengine.utils import track_iter_progress, track_parallel_progress
from mmcv.ops.nms import batched_nms
from mmengine.visualization import Visualizer

from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from utils import filter_and_update_tracks
from masa.visualization.visualizer import random_color

def startup_masa(masa_config, masa_checkpoint, device="cuda:0", unified=True, det_config=None, det_checkpoint=None):
    masa_model = init_masa(masa_config, masa_checkpoint, device=device)
    masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
    if not unified:
        det_model = init_detector(det_config, det_checkpoint, palette="random", device=device)
        det_model.cfg.test_dataloader.dataset.pipeline[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)
    else:
        det_model = None
        test_pipeline = None
    
    return masa_model, masa_test_pipeline, det_model, test_pipeline


def detect_and_track_with_roi(video_reader, masa_model, masa_test_pipeline, texts="person", unified=True, test_pipeline=None, det_model=None, no_post=False, fp16=False):
    frame_idx = 0
    instances_list = []
    frames = []
    
    for frame in track_iter_progress((video_reader, len(video_reader))):
        def hook_fn(module, input, output):
                # Save the output in an attribute, so you can retrieve it later
                module.feature_map = output

            # Register the hook on the neck (ChannelMapper) of the detector:
        masa_model.detector.neck.register_forward_hook(hook_fn)
        if unified:
            # Perform the base inference using MASA
            track_result = inference_masa(masa_model, frame,
                                        frame_id=frame_idx,
                                        video_len=len(video_reader),
                                        test_pipeline=masa_test_pipeline,
                                        text_prompt=texts,
                                        fp16=fp16)
            # ----- ROI-Align Integration for the unified branch -----
            feature_map = masa_model.detector.neck.feature_map
            # Original input resolution
            H, W = frame.shape[:2]

            # Feature maps from the neck (hooked earlier)
            feature_maps = masa_model.detector.neck.feature_map

            for i, fmap in enumerate(feature_maps):
                B, C, Hf, Wf = fmap.shape
                scale_H = H / Hf
                scale_W = W / Wf

            # If the output is a tuple, get the first tensor.
            if isinstance(feature_map, (tuple, list)):
                feature_map = feature_map[0]
                
            
            # Extract detected bounding boxes
            det_bboxes = track_result[0].pred_track_instances.bboxes  # Expected to be [N, 4]
            batch_boxes = []
            # Prepare boxes with batch index (assuming all boxes belong to batch index 0)
            for bbox in det_bboxes:
                x1, y1, x2, y2 = bbox.tolist()
                batch_boxes.append([0, x1, y1, x2, y2])
            if len(batch_boxes) > 0:
                batch_boxes = torch.tensor(batch_boxes, dtype=torch.float, device=feature_map.device)
                # Use output_size (7,7) and spatial_scale (1/16) as an example
                roi_features = roi_align(feature_map, batch_boxes, output_size=(7, 7), spatial_scale=1/16.0)
            else:
                roi_features = None
            # Attach ROI features to the tracking result
            track_result[0].roi_features = roi_features

        
        else:
            # Detection branch processing
            result = inference_detector(det_model, frame,
                                        text_prompt=texts,
                                        test_pipeline=test_pipeline,
                                        fp16=fp16)
            det_bboxes, keep_idx = batched_nms(
                boxes=result.pred_instances.bboxes,
                scores=result.pred_instances.scores,
                idxs=result.pred_instances.labels,
                class_agnostic=True,
                nms_cfg=dict(type="nms",
                             iou_threshold=0.5,
                             class_agnostic=True,
                             split_thr=100000))
            det_bboxes = torch.cat([det_bboxes, result.pred_instances.scores[keep_idx].unsqueeze(1)], dim=1)
            det_labels = result.pred_instances.labels[keep_idx]
            # Pass detection results into inference_masa
            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          det_bboxes=det_bboxes,
                                          det_labels=det_labels,
                                          fp16=fp16)
            # (You can also include ROI-Align here if desired in a similar fashion.)
        
        frame_idx += 1
        print('Number of bbox detected:', len(track_result[0].pred_track_instances.bboxes))
        
        # Make sure bboxes are in float32
        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to("cpu"))
        frames.append(frame)
    
    if not no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))
    
    return instances_list, frames



def create_tracks(instances_list, score_threshold=0.2):
    track_data = {}
    
    for frame_idx, track_result in enumerate(instances_list):
        data_sample = track_result[0]

        if "pred_track_instances" in data_sample:
            pred_instances = data_sample.pred_track_instances
        
            if "scores" in pred_instances:
                pred_instances = pred_instances[pred_instances.scores > score_threshold]
        
            if ("instances_id" in pred_instances) and (pred_instances.instances_id.size()):
                for _id, (x1, y1, x2, y2) in zip(pred_instances.instances_id, pred_instances.bboxes):
                    x1, y1 = x1.item(), y1.item()
                    x2, y2 = x2.item(), y2.item()
                    r_id = _id.item()
                    new_bb = (x1, y1, x2, y2)
                    new_point = ((x1 + x2) / 2, max(y1, y2))
                    
                    if r_id not in track_data:
                        frame_data = (new_point, (0.0, 0.0), (0.0, 0.0), new_bb)
                        track_data[r_id] = (random_color(_id), {frame_idx : frame_data})

                    else:
                        frames_data = track_data[r_id][1]
                        last_point = frames_data[next(reversed(frames_data.keys()))][0]
                        new_x_couple = (last_point[0], new_point[0])
                        new_y_couple = (last_point[1], new_point[1])
                        frames_data[frame_idx] = (new_point, new_x_couple, new_y_couple, new_bb)

    return track_data

def serialize_track_data(f_name, track_data):
    with open(f_name, "w") as f:
        json.dump(track_data, f, indent=4)

def deserialize_track_data(f_name):
    with open(f_name, "r") as f:
        return json.load(f)

def draw_frame(visualizer, track_data, frame, frame_idx, track_width=5, bb_width=10, bb_alpha=0.6, bb_text_size=None, no_track=False, no_bb=False):
    image = frame[:, :, ::-1]

    visualizer.set_image(image)
    for id, (color, frames_data) in track_data.items():
        idx = 0
        x_c_lists = [[]]
        y_c_lists = [[]]
        bb_found = False
        for f_idx, (point, x_couple, y_couple, f_bb) in frames_data.items():
            if int(f_idx) > frame_idx:
                break
            if (x_couple != (0, 0) or y_couple != (0, 0)):
                x_c_lists[idx].append(x_couple)
                y_c_lists[idx].append(y_couple)
            else:
                idx += 1
                x_c_lists.append([])
                y_c_lists.append([])
            if int(f_idx) == frame_idx:
                bb = f_bb
                bb_found = True
                break
        x_c_lists = [x_couple for x_couple in x_c_lists if x_couple]
        y_c_lists = [y_couple for y_couple in y_c_lists if y_couple]

        if (not no_track) and (x_c_lists):
            for x_couples, y_couples in zip(x_c_lists, y_c_lists):
                    visualizer.draw_lines(x_datas=torch.tensor(x_couples), y_datas=torch.tensor(y_couples), colors=tuple(color), line_widths=track_width)
        if (not no_bb) and (bb_found):
            visualizer.draw_bboxes(bboxes=torch.tensor(bb), edge_colors=tuple(color), line_widths=bb_width, alpha=bb_alpha)
            visualizer.draw_texts(texts=str(id), positions=torch.tensor((bb[0], bb[1])), font_sizes=bb_text_size)
    drawn_img = visualizer.get_image()
    
    gc.collect()
    return drawn_img

def draw_frame_unpacker(arg):
    return draw_frame(*arg)

def draw_video(frames, track_data, track_width=5, bb_width=10, bb_alpha=0.6, bb_text_size=None, disable_track_video=False, disable_bb=False):
    visualizer = Visualizer()
    num_cores = max(1, min(os.cpu_count() - 1, 16))
    frames_per_core = min((len(frames) // num_cores) + 1, 32)

    frames = track_parallel_progress(
                    draw_frame_unpacker,
                    [(visualizer, track_data, frame, idx, track_width, bb_width, bb_alpha, bb_text_size, disable_track_video, disable_bb) for idx, frame in enumerate(frames)],
                    num_cores,
                    chunksize=frames_per_core)

    return frames

def write_video(out_video, frames, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    for frame in track_iter_progress(frames, len(frames)):
        video_writer.write(frame[:, :, ::-1])
    video_writer.release()

def parse_args():
    parser = argparse.ArgumentParser(prog="People Tracker")
    
    parser.add_argument("--detect", action="store_true", help="Use detection mode")
    parser.add_argument("--masa_config", help="Masa Config file")
    parser.add_argument("--masa_checkpoint", help="Masa Checkpoint file")
    parser.add_argument("--det_config", help="Detector Config file")
    parser.add_argument("--det_checkpoint", help="Detector Checkpoint file")
    parser.add_argument("--unified", action="store_true", help="Use unified model, which means the masa adapter is built upon the detector model")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--score-thr", type=float, default=0.2, help="Bbox score threshold")
    parser.add_argument("--texts", type=str, default="person", help="Text prompt")
    parser.add_argument("--fp16", action="store_true", help="Activate fp16 mode")
    parser.add_argument("--no-post", action="store_true", help="Do not post-process the results")
    parser.add_argument("--disable_out_video", action="store_true", help="Disable drawing video, works only in detect mode")
    parser.add_argument("--out_tracks", type=str, help="Output track file")
    parser.add_argument("--disable_track_file", action="store_true", help="Disable writing tracks file")

    parser.add_argument("--draw", action="store_true", help="Use draw mode")
    parser.add_argument("--in_tracks", type=str, help="Input track file")
    parser.add_argument("--individually", action="store_true", help="Draw every track in its video")
    parser.add_argument("--tracks_to_draw", nargs="*", help="Specify tracks to draw. Use -1 as a separator for group of tracks to render in a separate video")

    parser.add_argument("--in_video", type=str, help="Input video file")
    parser.add_argument("--out_video", type=str, help="Output video file")
    parser.add_argument("--track_width", type=int, default=5, help="Track width")
    parser.add_argument("--bb_width", type=int, default=10, help="Bounding boxes width")
    parser.add_argument("--bb_alpha", type=float, default=0.6, help="Bounding boxes alpha")
    parser.add_argument("--bb_text_size", type=int, default=None, help="Bounding boxes text size")
    parser.add_argument("--disable_track_video", action="store_true", help="Disable drawing tracks on video")
    parser.add_argument("--disable_bb", action="store_true", help="Disable drawing bounding boxes")
    
    return parser.parse_args()

def check_args(args):
    if not args.in_video:
        print("You have to specify on which video you want to work (--in_video)")
        exit()

    if args.detect:
        if not args.masa_config:
                print("You have to specify the masa configuration file (--masa_config)")
                exit()
        if (not args.unified) and (not args.det_config):
                print("In not unified mode, you have to specify the detector configuration file (--det_config)")
                exit()
        if (not args.disable_out_video) and (not args.out_video):
            print("You have to specify where to store the output video (--out_video)")
            exit()
        if (not args.disable_track_file) and (not args.out_tracks):
            print("You have to specify where to store the output tracks file (--out_tracks)")
            exit()

    elif args.draw:
        if not args.in_tracks:
            print("You have to specify from which file to read the tracks (--in_tracks)")
            exit()
        if not args.out_video:
            print("You have to specify where to store the output video (--out_video)")
            exit()

    else:
        print("When running the script either use detect (--detect) or draw (--draw) mode")
        exit()

def detect_mode(args):
    print("Setting up...")
    masa_model, masa_test_pipeline, det_model, test_pipeline = startup_masa(args.masa_config,
                                                                            args.masa_checkpoint,
                                                                            args.device,
                                                                            args.unified,
                                                                            args.det_config,
                                                                            args.det_checkpoint)
    video_reader = mmcv.VideoReader(args.in_video)

    print("Starting to detect and track...")
    instances_list, frames = detect_and_track_with_roi(video_reader,
                                    masa_model,
                                    masa_test_pipeline,
                                    args.texts,
                                    args.unified,
                                    test_pipeline,
                                    det_model,
                                    args.no_post,
                                    args.fp16)
    #track_data = create_tracks(instances_list, args.score_thr)

    #if not args.disable_track_file:
        #print("Serializing track data...")
        #serialize_track_data(args.out_tracks, track_data)

    #if not args.disable_out_video:
        #print("Starting to draw...")
        #frames = draw_video(frames, track_data, args.track_width, args.bb_width, args.bb_alpha, args.bb_text_size, args.disable_track_video, args.disable_bb)
        
        #print("Starting to write...")
        #write_video(args.out_video, frames, video_reader.fps, video_reader.width, video_reader.height)
        
        #print("Done")

def draw_mode(args):
    print("Deserializing track data...")
    track_data = deserialize_track_data(args.in_tracks)
    video_reader = mmcv.VideoReader(args.in_video)

    if args.individually:
        for s_track_id, s_track_data in track_data.items():
            print(f"Starting to draw {s_track_id}th track image...")
            frames = draw_video(video_reader[:], {s_track_id : s_track_data}, args.track_width, args.bb_width, args.bb_alpha, args.bb_text_size, args.disable_track_video, args.disable_bb)
            
            print(f"Starting to write {s_track_id}th track image...")
            out_video = os.path.splitext(args.out_video)[0] + "_" + str(s_track_id) + ".mp4"
            write_video(out_video, frames, video_reader.fps, video_reader.width, video_reader.height)
    
    elif args.tracks_to_draw:
        tracks_to_draw = [{}]
        elem_idx = 0
        for track_id in args.tracks_to_draw:
            if track_id != "-1":
                tracks_to_draw[elem_idx][track_id] = track_data[track_id]
            else:
                elem_idx = elem_idx + 1
                tracks_to_draw.append({})
        
        for idx, tracks in enumerate(tracks_to_draw):
            print(f"Starting to draw {idx + 1}th image...")
            frames = draw_video(video_reader[:], tracks, args.track_width, args.bb_width, args.bb_alpha, args.bb_text_size, args.disable_track_video, args.disable_bb)
            
            print(f"Starting to write {idx + 1}th image...")
            out_video = os.path.splitext(args.out_video)[0] + "_" + str(idx) + ".mp4"
            write_video(out_video, frames, video_reader.fps, video_reader.width, video_reader.height)
    
    else:
        print("Starting to draw...")
        frames = draw_video(video_reader[:], track_data, args.track_width, args.bb_width, args.bb_alpha, args.bb_text_size, args.disable_track_video, args.disable_bb)

        print("Starting to write...")
        write_video(args.out_video, frames, video_reader.fps, video_reader.width, video_reader.height)
    
    print("Done")

def main():
    args = parse_args()
    check_args(args)
    seed = 42

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if args.detect:
        detect_mode(args)
    elif args.draw:
        draw_mode(args)

if __name__ == "__main__":
    main()