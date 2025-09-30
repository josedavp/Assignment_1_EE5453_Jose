EE 5453 Class assignment. 

To verify GPU usage:
python check_gpu.py

##To train the model:
python train.py --epochs 7 --batch_size 64 --block_size 128


To generate response:
python generate.py --prompt "ROMEO:" --tokens 200 --out sample_romeo.txt
python generate.py --prompt "JULIET:" --tokens 200 --out sample_juliet.txt
