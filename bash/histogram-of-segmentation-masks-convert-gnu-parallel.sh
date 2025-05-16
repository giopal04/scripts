#!/bin/bash

find . -type f -name '*.png' | parallel -j 16 "echo {} | histogram-of-segmentation-masks-convert.sh" | awk '{for (i=1 ; i<=NF; i++) printf("%s ", $i) ; print("")}'

exit 0
