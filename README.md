# Progetto-IVA
progetto per laboratorio di Image and Video Analysis

# How to run this project
0) git clone of this repo:
   
   ```
   git clone https://github.com/gianlu9911/Progetto-IVA.git
   ```
2) Very important: create a folder called 'saved_models'. Make sure to to download the models from here:
   ```[https://github.com/siyuanliii/masa?tab=readme-ov-file](https://huggingface.co/dereksiyuanli/masa/resolve/main/gdino_masa.pth)```
3) Please, create a virtual environment. With Conda you can run:
   conda create masaenv
   Tested with python 11 working.
5) Remember to activate it! you can run: conda activate masaenv
6) In order to fulfill every requirements, plese, use the install_dependecies.bat or install_dependecies.sh

# Feature Extraction
Once you have the virtual environment with the requirements and the folder with the model you can run:
python my_scripts\featureMeanStd.py --detect --in_video videos\ins\clip_0.mp4 --out_video videos\outs\micc_2_out_unified.mp4 --out_tracks tracks\micc_2_tracks_unified.json --unified --masa_config configs\masa-gdino\masa_gdino_swinb_inference.py --masa_checkpoint saved_models\masa_models\gdino_masa.pth

