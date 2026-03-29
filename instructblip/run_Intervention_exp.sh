# !/bin/bash
# This script runs the experiments for different intervention layers and saves the results in the corresponding text files.
mkdir -p Intervention_exp
python instructblip7B_exp.py --intervention_strength 0.0 --intervention_layer l:31 > Intervention_exp/0.txt &&
python instructblip7B_exp.py --intervention_strength 0.1 --intervention_layer l:31 > Intervention_exp/1.txt &&
python instructblip7B_exp.py --intervention_strength 0.2 --intervention_layer l:31 > Intervention_exp/2.txt &&
python instructblip7B_exp.py --intervention_strength 0.3 --intervention_layer l:31 > Intervention_exp/3.txt &&
python instructblip7B_exp.py --intervention_strength 0.4 --intervention_layer l:31 > Intervention_exp/4.txt &&
python instructblip7B_exp.py --intervention_strength 0.5 --intervention_layer l:31 > Intervention_exp/5.txt &&
python instructblip7B_exp.py --intervention_strength 0.6 --intervention_layer l:31 > Intervention_exp/6.txt &&
python instructblip7B_exp.py --intervention_strength 0.7 --intervention_layer l:31 > Intervention_exp/7.txt &&
python instructblip7B_exp.py --intervention_strength 0.8 --intervention_layer l:31 > Intervention_exp/8.txt &&
python instructblip7B_exp.py --intervention_strength 0.9 --intervention_layer l:31 > Intervention_exp/9.txt &&
python instructblip7B_exp.py --intervention_strength 1.0 --intervention_layer l:31 > Intervention_exp/10.txt