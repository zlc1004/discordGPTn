conda create -p .conda python==3.11 -y
conda activate ./.conda
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
