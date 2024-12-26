#!/bin/sh

run_experiment() {
        task=$1
        model=$2

        echo model=$model task=$task

        logprefix="save/$model/$task"

        for seed in {10,20,40}
        do
                for patient in {0001,0002,0003,0005,0006,0007,1000,1001,1002,1003,1004,1101,1102}
                do 
                        #if [ ! -f "$logprefix/logs/$patient-$seed.log" ]
                        #then
                                while [ "$(pgrep -c -P$$)" -ge 10 ]; do sleep 1; done

                                echo seed=$seed patient=$patient
                                
                                python3 main.py --patient $patient \
                                                --seed $seed \
                                                --task $task \
                                                --input-type $model \
                                                --num-epochs 500 \
                                                --batch-size 8 \
                                                --logprefix $logprefix &
                                                
                        #fi
                done
        done

        while [ "$(pgrep -c -P$$)" -ge 1 ]; do sleep 1; done
        python3 ../../parse_logs.py --prefix $logprefix
}

#run_experiment 'regression' 'kinematic'
#run_experiment 'regression' 'power-spectral-difference'
run_experiment 'regression' 'power-spectral-no-eeg'
run_experiment 'regression' 'power-spectral-no-kinematic'
