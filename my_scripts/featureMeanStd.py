import os
import sys

import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import nltk
nltk.download('averaged_perceptron_tagger_eng')
import warnings
warnings.filterwarnings("ignore")
import argparse
from torchvision.ops import roi_align
import torch
import re
from collections import defaultdict
import mmcv
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmengine.utils import track_iter_progress
from mmcv.ops.nms import batched_nms
from mmengine.visualization import Visualizer

from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline


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
    

def detect_and_track_with_roi(video_reader, masa_model, masa_test_pipeline, texts="person",
                              unified=True, test_pipeline=None, det_model=None,
                              no_post=False, fp16=False):

    frame_idx = 0
    features_dir = "saved_features_per_neck"
    os.makedirs(features_dir, exist_ok=True)
    score_threshold = 0.5  

    # -------------------
    # 1. Hook sul neck UNA SOLA VOLTA
    # -------------------
    def hook_fn(module, input, output):
        module.feature_map = output
    masa_model.detector.neck.register_forward_hook(hook_fn)

    for frame in track_iter_progress((video_reader, len(video_reader))):

        if unified:
            # Inference con MASA
            track_result = inference_masa(
                masa_model,
                frame,
                frame_id=frame_idx,
                video_len=len(video_reader),
                test_pipeline=masa_test_pipeline,
                text_prompt=texts,
                fp16=fp16
            )

            # Recupero features dal neck
            feature_map = masa_model.detector.neck.feature_map
            if isinstance(feature_map, (tuple, list)):
                feature_map = feature_map[0]  # primo livello se multilivello

            _, _, H_feat, W_feat = feature_map.shape
            H, W = frame.shape[:2]

            # Bounding boxes, score e ID
            det_instances = track_result[0].pred_track_instances
            det_bboxes = det_instances.bboxes
            scores = det_instances.scores
            track_ids = det_instances.instances_id

            #print(f"Frame {frame_idx}: {len(det_bboxes)} detections")
            #print(f"Scores: {scores.tolist()}")

            # ROI Align
            batch_boxes = []
            for bbox in det_bboxes:
                x1, y1, x2, y2 = bbox.tolist()
                batch_boxes.append([0, x1, y1, x2, y2])

            if batch_boxes:
                batch_boxes_tensor = torch.tensor(batch_boxes, dtype=torch.float, device=feature_map.device)
                spatial_scale = W_feat / W  # attenzione: valido se resize lineare

                roi_feats = roi_align(
                    feature_map,
                    batch_boxes_tensor,
                    output_size=(7, 7),
                    spatial_scale=spatial_scale
                )

                # Salvataggio features filtrate per score
                for bbox, tid, roi_feat, score in zip(det_bboxes, track_ids, roi_feats, scores):
                    if score >= score_threshold:
                        roi_feat_np = roi_feat.cpu().numpy()
                        x1, y1, x2, y2 = bbox.tolist()
                        fname = f"frame_{frame_idx:05d}_id_{int(tid)}_bbox_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.npy"
                        np.save(os.path.join(features_dir, fname), roi_feat_np)

        else:
            print("Unified mode required for ROI features.")
        
        frame_idx += 1





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


def main():
    args = parse_args()
    check_args(args)
    
    if args.detect:
        detect_mode(args)
    elif args.draw:
        print("not implemented")

if __name__ == "__main__":
    main()


