#!/bin/bash
for i in {1..64}
do
    echo $i
    python PredictGoTerm_qsub.py ${i} 64 > output_${i}.txt &
done
