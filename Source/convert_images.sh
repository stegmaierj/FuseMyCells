#!/bin/bash

input_file=$1
output_path=$2

MYCOMMAND='/netshares/scratch/stegmaier/Software/Environments/fusemycells/bin/python /netshares/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/convert_images.py '$input_file' '$output_path

echo $MYCOMMAND
eval $MYCOMMAND