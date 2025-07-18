# Progetto-IVA
progetto per laboratorio di Image and Video Analysis

# How to run this project
0) git clone of this repo:
   
   ```
   git clone https://github.com/gianlu9911/Progetto-IVA.git
   ```
2) Very important: create a folder called 'saved_models'. Make sure to to download the models from here:
   ```https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth```
3) Please, create a virtual environment. With Conda you can run:
   ```
   conda create masaenv
   ```
   Tested with python 11 working.
5) Activate it
   ```
   conda activate masaenv
   ```
7) In order to fulfill every requirements, plese, use the ```install_dependecies.bat``` or ```install_dependecies.sh```

# Feature Extraction
Once you have the virtual environment with the requirements and the folder with the model you can run:
```
python my_scripts\featureMeanStd.py --detect --in_video videos\ins\clip_0.mp4 --out_video videos\outs\micc_2_out_unified.mp4 --out_tracks tracks\micc_2_tracks_unified.json --unified --masa_config configs\masa-gdino\masa_gdino_swinb_inference.py --masa_checkpoint saved_models\masa_models\gdino_masa.pth
```

A folder called ```saved_features_per_bbox``` will be generated. It containes the ROI features, for each frame, of every id over a certain bounding box.

# Gaussian Mixtures

REQUIRES to execute the featureMeanStd.py script above!

run ```my_scripts/exploration_features.py``` script to fit a Guassian Mixture over the feature


