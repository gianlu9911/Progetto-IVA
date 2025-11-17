# Aggiorna pip e installa openmim
pip install --no-cache-dir --force-reinstall -U pip
pip install --no-cache-dir --force-reinstall -U openmim

# Installazioni con MIM
pip install --no-cache-dir --force-reinstall mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install --no-cache-dir --force-reinstall git+https://github.com/open-mmlab/mmdetection.git@v3.3.0

# Pacchetti Python generici
pip install --no-cache-dir --force-reinstall supervision
pip install --no-cache-dir --force-reinstall transformers==4.38.2
pip install --no-cache-dir --force-reinstall nltk==3.8.1
pip install --no-cache-dir --force-reinstall h5py
pip install --no-cache-dir --force-reinstall einops
pip install --no-cache-dir --force-reinstall seaborn
pip install --no-cache-dir --force-reinstall fairscale
pip install --no-cache-dir --force-reinstall git+https://github.com/openai/CLIP.git --no-deps
pip install --no-cache-dir --force-reinstall git+https://github.com/siyuanliii/TrackEval.git
pip install --no-cache-dir --force-reinstall git+https://github.com/SysCV/tet.git#subdirectory=teta
pip install --no-cache-dir --force-reinstall git+https://github.com/scalabel/scalabel.git@scalabel-evalAPI
pip install --no-cache-dir --force-reinstall git+https://github.com/TAO-Dataset/tao
pip install --no-cache-dir --force-reinstall git+https://github.com/lvis-dataset/lvis-api.git

# OpenCV e numpy (versione iniziale)
pip install --no-cache-dir --force-reinstall opencv-python==4.11.0.86
pip install --no-cache-dir --force-reinstall numpy==1.26.4

# PyTorch e moduli correlati (installati alla fine)
pip install --no-cache-dir --force-reinstall torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# numpy versione finale
pip install --no-cache-dir --force-reinstall numpy==1.26.4
pip install --no-cache-dir --force-reinstall mdet
