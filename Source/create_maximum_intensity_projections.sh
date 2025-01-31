#!/bin/bash

input_file=$1
output_path=$2

echo $input_file
echo $output_path

MYCOMMAND='/netshares/scratch/stegmaier/Software/Environments/fusemycells/bin/python /netshares/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/create_maximum_intensity_projections.py --input_file='$input_file' --output_path='$output_path

echo $MYCOMMAND
eval $MYCOMMAND
