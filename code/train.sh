# 1️initialize conda

source /root/miniconda3/bin/conda/conda.sh

# 2️create env and activate
conda create -n faster python=3.10 -y
conda activate faster

# 3️install packages

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python tqdm matplotlib numpy<2
pip install ultralytics

# 4️train

cd /root/autodl-tmp/fast_r_cnn
python /root/autodl-tmp/fast_r_cnn/faster_r_cnn_marked.py \
  --data_root /root/autodl-tmp/dataset \
  --eval_map --eval_prf1 --eval_prauc \

  > /root/autodl-tmp/fast_r_cnn/train_output.log 2>&1