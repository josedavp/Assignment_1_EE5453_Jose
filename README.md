# EE 5453 Class assignment. 

### To verify GPU usage: 
```bash
python check_gpu.py
```

## To train the model: 
```bash
python train.py --epochs 7 --batch_size 64 --block_size 128
```

### To generate responses:
```bash
python generate.py --prompt "ROMEO:" --tokens 200 --out sample_romeo.txt \n
python generate.py --prompt "JULIET:" --tokens 200 --out sample_juliet.txt
```

#### File outputs are found in folder 'out'
