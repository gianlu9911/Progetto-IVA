# Progetto-IVA
progetto per laboratorio di Image and Video Analysis

# How to run this project

0) 
Make sure you have conda installed on your system, after that clone the repo
   ```
   git clone https://github.com/gianlu9911/Progetto-IVA.git
   ```

1)
   ```
   conda create -n masaenv python=3.11 -y
   conda activate masaenv
   cd Progetto-IVA
   chmod +x install_dependencies.sh
   sh install_dependencies.sh

   eventually, if required

   pip3 install -U scikit-learn
   pip install mmdet
   ```

2)

```
mkdir saved_models
cd saved_models
mkdir masa_models
cd masa_models
wget https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth
```



# Feature Extraction
Once you have the virtual environment with the requirements and the folder with the model you can run:
```
python my_scripts/featureMeanStd.py --detect --in_video videos/ins/clip_0.mp4 --out_video videos/outs/micc_2_out_unified.mp4 --out_tracks tracks/micc_2_tracks_unified.json --unified --masa_config configs/masa-gdino/masa_gdino_swinb_inference.py --masa_checkpoint saved_models/masa_models/gdino_masa.pth
```

A folder called ```saved_features_per_bbox``` will be generated. It containes the ROI features, for each frame, of every id over a certain bounding box.

# Gaussian Mixtures

REQUIRES to execute the featureMeanStd.py script above!

```
python my_scripts/heuristic.py --feature_dir saved_features_per_neck
```
Run script to execute the heuristic for id re-assignments.


