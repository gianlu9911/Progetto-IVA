ECHO OFF

REM Update pip and install openmim
ECHO "Updating pip and installing openmim..."
python -m pip install -U pip || GOTO :error
pip install -U openmim || GOTO :error

REM Install packages using MIM
ECHO "Installing packages with MIM..."
mim install mmengine || GOTO :error
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html || GOTO :error
pip install git+https://github.com/open-mmlab/mmdetection.git@v3.3.0 || GOTO :error

REM Install various Python packages using pip
ECHO "Installing various Python packages..."
pip install numpy==1.26.4 || GOTO :error
pip install supervision || GOTO :error
pip install transformers==4.38.2 || GOTO :error
pip install nltk==3.8.1 || GOTO :error
pip install h5py || GOTO :error
pip install einops || GOTO :error
pip install seaborn || GOTO :error
pip install fairscale || GOTO :error
pip install git+https://github.com/openai/CLIP.git --no-deps || GOTO :error
pip install git+https://github.com/siyuanliii/TrackEval.git || GOTO :error
pip install git+https://github.com/SysCV/tet.git#subdirectory=teta || GOTO :error
pip install git+https://github.com/scalabel/scalabel.git@scalabel-evalAPI || GOTO :error
pip install git+https://github.com/TAO-Dataset/tao || GOTO :error
pip install git+https://github.com/lvis-dataset/lvis-api.git || GOTO :error

ECHO "All packages installed successfully!"
GOTO :end

:error
ECHO "Encountered an error, exiting..."

:end