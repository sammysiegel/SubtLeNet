#!/bin/bash 

for ntrunc in 5 6; do 
    for nlimit in 100; do 
        echo 'TRAINING' baseline $ntrunc $nlimit
        source ./setup.sh
        ./train_particles.py --nepoch 50 --trunc $ntrunc --limit $nlimit 
        ./infer_particles.py \
            --h5 models/particles/v4_Adam_trunc${ntrunc}_limit${nlimit}/baseline_best.h5 \
            --name baseline_Adam_${ntrunc}_${nlimit}
    done
done

exit 1

for ntrunc in 7; do
    for v in finegrid noetaphi etaphi; do
        for nlimit in 100; do 
            echo 'TRAINING' $v $ntrunc $nlimit
            source ./setup_${v}.sh
            ./train_scenarios.py --nepoch 50 --trunc $ntrunc --limit $nlimit --version 4_$v
            ./infer_particles.py \
                --h5 models/particles/v4_${v}_trunc${ntrunc}_limit${nlimit}/baseline_best.h5 \
                --name baseline_${ntrunc}_${nlimit} 
        done
    done
done
    

