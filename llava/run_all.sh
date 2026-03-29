# !/bin/bash

# This script runs all the experiments for LLaVa7B and saves the results in the corresponding text files.
mkdir -p LLaVA7b
python LLaVA7B_exp.py --intervention_layer v:0 > LLaVA7b/v0.txt &&
python LLaVA7B_exp.py --intervention_layer v:4 > LLaVA7b/v4.txt &&
python LLaVA7B_exp.py --intervention_layer v:8 > LLaVA7b/v8.txt &&
python LLaVA7B_exp.py --intervention_layer v:12 > LLaVA7b/v12.txt &&
python LLaVA7B_exp.py --intervention_layer v:15 > LLaVA7b/v15.txt &&
python LLaVA7B_exp.py --intervention_layer v:23 > LLaVA7b/v23.txt
python LLaVA7B_exp.py --intervention_layer l:0 > LLaVA7b/l0.txt &&
python LLaVA7B_exp.py --intervention_layer l:8 > LLaVA7b/l8.txt &&
python LLaVA7B_exp.py --intervention_layer l:16 > LLaVA7b/l16.txt &&
python LLaVA7B_exp.py --intervention_layer l:24 > LLaVA7b/l24.txt &&
python LLaVA7B_exp.py --intervention_layer l:31 > LLaVA7b/l31.txt &&

# Run all the experiments for LLaVa13B and save the results in the corresponding text files.
mkdir -p LLaVA13b
python LLaVA13B_exp.py --intervention_layer v:0 > LLaVA13b/v0.txt &&
python LLaVA13B_exp.py --intervention_layer v:4 > LLaVA13b/v4.txt &&
python LLaVA13B_exp.py --intervention_layer v:8 > LLaVA13b/v8.txt &&
python LLaVA13B_exp.py --intervention_layer v:12 > LLaVA13b/v12.txt &&
python LLaVA13B_exp.py --intervention_layer v:15 > LLaVA13b/v15.txt &&
python LLaVA13B_exp.py --intervention_layer v:23 > LLaVA13b/v23.txt
python LLaVA13B_exp.py --intervention_layer l:0 > LLaVA13b/l0.txt &&
python LLaVA13B_exp.py --intervention_layer l:8 > LLaVA13b/l8.txt &&
python LLaVA13B_exp.py --intervention_layer l:16 > LLaVA13b/l16.txt &&
python LLaVA13B_exp.py --intervention_layer l:24 > LLaVA13b/l24.txt &&
python LLaVA13B_exp.py --intervention_layer l:31 > LLaVA13b/l31.txt &&



# Run all the experiments for LLaVaNeXT7B and save the results in the LLaVaNeXT13b directory.
mkdir -p LLaVaNeXT7b
python LLaVaNeXT7B_exp.py --intervention_layer v:0 > LLaVaNeXT7b/v0.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer v:4 > LLaVaNeXT7b/v4.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer v:8 > LLaVaNeXT7b/v8.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer v:12 > LLaVaNeXT7b/v12.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer v:15 > LLaVaNeXT7b/v15.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer v:23 > LLaVaNeXT7b/v23.txt
python LLaVaNeXT7B_exp.py --intervention_layer l:0 > LLaVaNeXT7b/l0.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer l:8 > LLaVaNeXT7b/l8.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer l:16 > LLaVaNeXT7b/l16.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer l:24 > LLaVaNeXT7b/l24.txt &&
python LLaVaNeXT7B_exp.py --intervention_layer l:31 > LLaVaNeXT7b/l31.txt &&

# Run all the experiments for LLaVaNeXT13b and save the results in the corresponding text files.
mkdir -p LLaVaNeXT13b
python LLaVaNeXT13B_exp.py --intervention_layer v:0 > LLaVaNeXT13b/v0.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer v:4 > LLaVaNeXT13b/v4.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer v:8 > LLaVaNeXT13b/v8.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer v:12 > LLaVaNeXT13b/v12.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer v:15 > LLaVaNeXT13b/v15.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer v:23 > LLaVaNeXT13b/v23.txt
python LLaVaNeXT13B_exp.py --intervention_layer l:0 > LLaVaNeXT13b/l0.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer l:8 > LLaVaNeXT13b/l8.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer l:16 > LLaVaNeXT13b/l16.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer l:24 > LLaVaNeXT13b/l24.txt &&
python LLaVaNeXT13B_exp.py --intervention_layer l:31 > LLaVaNeXT13b/l31.txt


