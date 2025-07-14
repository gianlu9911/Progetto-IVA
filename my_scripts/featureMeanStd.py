import os
import sys

import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings("ignore")
import argparse
from torchvision.ops import roi_align
import torch


import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmengine.utils import track_iter_progress
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

def compute_mean_std(features_dir):
    """
    Compute the mean and standard deviation of features saved in the specified directory.
    """

    # ---- Configurations ----
    num_levels = 4
    num_frames = 144

    # ---- Loop over pyramid levels ----
    for level in range(num_levels):
        print(f"Processing dense features for level {level}...")

        all_features = []

        for frame_idx in range(num_frames):
            # Load dense feature file for this frame and level
            fname = f"features_dense_frame_{frame_idx:05d}_level_{level}.npy"
            fpath = os.path.join(features_dir, fname)

            if not os.path.exists(fpath):
                print(f"Warning: {fpath} not found.")
                continue

            fmap = np.load(fpath)  # shape: (1, C, H, W) or (C, H, W)
            if fmap.ndim == 4:
                fmap = fmap.squeeze(0)  # Remove batch dim → (C, H, W)

            C, H, W = fmap.shape
            fmap = fmap.reshape(C, -1).T  # shape: (H*W, C)
            all_features.append(fmap)

        if not all_features:
            print(f"No features found for level {level}. Skipping.")
            continue

        # Stack all features into one matrix → shape: (N_total, C)
        all_features_matrix = np.vstack(all_features)

        # Compute mean and covariance
        mean = np.mean(all_features_matrix, axis=0)  # shape: (C,)
        cov = np.cov(all_features_matrix, rowvar=False)  # shape: (C, C)

        # Save to disk
        mean_path = os.path.join(features_dir, f"mean_dense_level_{level}.npy")
        cov_path = os.path.join(features_dir, f"cov_dense_level_{level}.npy")

        np.save(mean_path, mean)
        np.save(cov_path, cov)

        print(f"Saved mean and covariance for level {level}.")

    print("Done.")




        
    


def detect_and_track_with_roi(video_reader, masa_model, masa_test_pipeline, texts="person", unified=True, test_pipeline=None, det_model=None, no_post=False, fp16=False):
    frame_idx = 0
    instances_list = []
    frames = []
    features_dir = "saved_features"
    os.makedirs(features_dir, exist_ok=True)

    

    
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
            dense_fm = masa_model.detector.neck.feature_map

            # Save all pyramid levels
            for level_idx, fmap in enumerate(dense_fm):
                fmap_np = fmap.cpu().numpy()
                fname = f"features_dense_frame_{frame_idx:05d}_level_{level_idx}.npy"
                np.save(os.path.join(features_dir, fname), fmap_np)

            

            for i, fmap in enumerate(feature_map):
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
                # Save ROI features if any
                if roi_features is not None:
                    roi_features_np = roi_features.cpu().numpy()
                    fname = f"features_roi_frame_{frame_idx:05d}.npy"
                    np.save(os.path.join(features_dir, fname), roi_features_np)
            else:
                roi_features = None
            # Attach ROI features to the tracking result
            track_result[0].roi_features = roi_features

        
        else:
            print("Using detection branch")
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
        #print('Number of bbox detected:', len(track_result[0].pred_track_instances.bboxes))
        
        # Make sure bboxes are in float32
      






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
    detect_and_track_with_roi(video_reader,
                                    masa_model,
                                    masa_test_pipeline,
                                    args.texts,
                                    args.unified,
                                    test_pipeline,
                                    det_model,
                                    args.no_post,
                                    args.fp16)
    compute_mean_std("saved_features")


def main():
    args = parse_args()
    check_args(args)
    
    if args.detect:
        detect_mode(args)

if __name__ == "__main__":
    main()
