#!/bin/bash

for i in `seq 1 100`
do
    echo $i
    python data_collect.py -H carla -p 2004 -T 1000 --skip-frames 0 -B $i
done