# !/bin/bash

# This script runs all the experiments for InstructBLIP7B and saves the results in the corresponding text files.
mkdir -p InstructBLIP7b
python instructblip7B_exp.py --intervention_layer v:0 > InstructBLIP7b/v0.txt &&
python instructblip7B_exp.py --intervention_layer v:4 > InstructBLIP7b/v4.txt &&
python instructblip7B_exp.py --intervention_layer v:8 > InstructBLIP7b/v8.txt &&
python instructblip7B_exp.py --intervention_layer v:12 > InstructBLIP7b/v12.txt &&
python instructblip7B_exp.py --intervention_layer v:15 > InstructBLIP7b/v15.txt &&
python instructblip7B_exp.py --intervention_layer v:23 > InstructBLIP7b/v23.txt
python instructblip7B_exp.py --intervention_layer l:0 > InstructBLIP7b/l0.txt &&
python instructblip7B_exp.py --intervention_layer l:8 > InstructBLIP7b/l8.txt &&
python instructblip7B_exp.py --intervention_layer l:16 > InstructBLIP7b/l16.txt &&
python instructblip7B_exp.py --intervention_layer l:24 > InstructBLIP7b/l24.txt &&
python instructblip7B_exp.py --intervention_layer l:31 > InstructBLIP7b/l31.txt &&

# Run all the experiments for InstructBLIP13B and save the results in the corresponding text files.
mkdir -p InstructBLIP13b
python instructblip13B_exp.py --intervention_layer v:0 > InstructBLIP13b/v0.txt &&
python instructblip13B_exp.py --intervention_layer v:4 > InstructBLIP13b/v4.txt &&
python instructblip13B_exp.py --intervention_layer v:8 > InstructBLIP13b/v8.txt &&
python instructblip13B_exp.py --intervention_layer v:12 > InstructBLIP13b/v12.txt &&
python instructblip13B_exp.py --intervention_layer v:15 > InstructBLIP13b/v15.txt &&
python instructblip13B_exp.py --intervention_layer v:23 > InstructBLIP13b/v23.txt
python instructblip13B_exp.py --intervention_layer l:0 > InstructBLIP13b/l0.txt &&
python instructblip13B_exp.py --intervention_layer l:8 > InstructBLIP13b/l8.txt &&
python instructblip13B_exp.py --intervention_layer l:16 > InstructBLIP13b/l16.txt &&
python instructblip13B_exp.py --intervention_layer l:24 > InstructBLIP13b/l24.txt &&
python instructblip13B_exp.py --intervention_layer l:31 > InstructBLIP13b/l31.txt