#!/bin/sh

#python main.py --patient 0003 --seed 10 --num-epochs 5 --batch-size 8 --input-type power-spectral-no-eeg --task regression --output --no-load-model --logprefix ./foo        #2> tests/baseline-classification.log
#python main.py --patient 0003 --seed 10 --num-epochs 5 --batch-size 8 --input-type power-spectral-difference --task regression --output             #2> tests/baseline-classification.log
#python main.py --patient 0001 --seed 10 --num-epochs 5 --input-type baseline --task regression --output                 2> tests/baseline-regression.log
#python main.py --patient 0001 --seed 10 --num-epochs 5 --input-type multi-segment --task classification --output        2> tests/multi-segment-classification.log
#python main.py --patient 0001 --seed 10 --num-epochs 5 --input-type multi-segment --task regression --output            2> tests/multi-segment-regression.log


#reproduction code.
python main.py --patient 0002 --seed 10 --task regression --input-type power-spectral-no-kinematic --num-epochs 5 --batch-size 8 --logprefix runs/exp1 --output --no-load-model --acc-threshold 0.20 --acc-neighborhood 1 --no-cuda