# EE 5453 Class assignment. This repository contains a implementation of a mini Transformer (Tiny Shakespeare) 
*AI (ChatGPT) was used in setup troubleshooting, debugging Anaconda/Python configuration errors (PyTorch through Anaconda is not compatible with 5080 just yet. I learned how to properly configure and debug virtual environments, understand how CUDA errors are formed, verify how environments are interconnected, and verify deeper knowledge on my GPU and system hardware. It helped me verify the structure of the README through best academic/industry practices. While it did provide guidance I verified each type and confirmed the overall functionality and system design. It also helped in explaining the research paper from a more technical form to easier to digest version after I read the paper first. It provided explanations on the Transformer code, and helped explain line my line while creating a template TODO for me to work on. I verified each step, reviewed each line, and saw how a Transformer is correctly processed.*

Primary Queries with Reflection, and Raw Queries are within the main repository:  
- [AI_Queries_Reflection](AI_Queries_Reflection.txt)  
- [raw_AI_queries](raw_AI_queries.txt)  


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
Hardware was tested with RTX 3060 Ti with CUDA 12.4

## Results Sumamry:
The model was trainged on the Tiny Shakespeare dataset it relied on 7 epochs with a block size of 128 and batch size of 64. It produced a stead decrease in validation and training losses across epochs from 1 to 0.4841 in Epoche 7. Validation loss remained at a steady '1.5060'. The model generated text samples for Romeo and Juliet in Shakespearean language. 

*Screenshot of training are shown in the images/ folder.*


