# Progetto-IVA
progetto per laboratorio di Image and Video Analysis

# How to run this project
FOR WINDOWS USERS
0) git clone of this repo:
   
   ```
   git clone https://github.com/gianlu9911/Progetto-IVA.git
   ```
2) Very important: create a folder called 'saved_models/masa_models' where to store your models. Make sure to to download the models from here:
   ```https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth```
3) Create a virtual environment. With Conda you can run:
   ```
   conda create -n masaenv
   ```
   Tested with python 11 working.
5) Activate it
   ```
   conda activate masaenv
   ```
7) In order to fulfill every requirements, plese, use the ```install_dependecies.bat``` or ```install_dependecies.sh```


# FOR LINUX USERS:
0) 
Make sure you have conda installed on your system, after that clone the repo
   ```
   git clone https://github.com/gianlu9911/Progetto-IVA.git
   ```

1)
   ```
   conda create -n masaenv python=3.11 -y
   conda activate masaenv
   ```

2)
   ```
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```

3)
   ```
   pip install numpy==1.26
   ```
4)
   ```
   pip install opencv-python==4.11.0.86
   ```
5)
   ```
   cd Progetto-IVA
   sh install_dependencies.sh
   ```

6)

```
mkdir saved_models
cd saved_models
mkdir masa_models
cd masa_models
wget https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth
```

7)

```
pip install -U scikit-learn
```


in case you are missing some nltk file, just add to featureMeanStd.py or any new python file and run it
```
import nltk
nltk.download('averaged_perceptron_tagger_eng')
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


