#!/bin/bash
inputFile="bunny.off"
inputDir="../models/"
input=$inputDir$inputFile
range=(1)



for i in "${range[@]}"
do
  echo "Running $input in $i"
  time ./Simplify $input 0.50 gpu $i 1
done 
