# EE 5453 Class assignment. This repository contains a implementation of a mini Transformer (Tiny Shakespeare) 

### Setup
Create and activate a conda environment (Python 3.11.13) was used.
```bash
conda create -n transformer python=3.11 -y
conda activate transformer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm
```

### To verify GPU usage: 
```bash
python check_gpu.py
```

## To train the model: 
```bash
python train.py --epochs 7 --batch_size 64 --block_size 128
```
### This produces:
* out/best_ckpt.pt (best checkpoint)
* out/training_log.csv (training/validation losses)
#### File outputs are written to 'out/' folder

### To generate responses:
```bash
python generate.py --prompt "ROMEO:" --tokens 200 --out sample_romeo.txt \n
python generate.py --prompt "JULIET:" --tokens 200 --out sample_juliet.txt
```

### Notes:
Code files needed: model.py, train.py, generate.py
Hardware was tested with RTX 3060 Ti
