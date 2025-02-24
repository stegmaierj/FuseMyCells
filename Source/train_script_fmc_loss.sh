#!/bin/bash

train_list=$1
val_list=$2
test_list=$3
output_path=$4
log_path=$5

surface_distance=$6
light_map=$7
in_channels=$8

echo $train_list
echo $val_list
echo $test_list
echo $output_path
echo $surface_distance
echo $light_map
echo $in_channels
echo $log_path

image_groups='data/raw_image'

if test $surface_distance -eq "1"
then
    image_groups=$image_groups' data/surface_distance'
fi
if test $light_map -eq "1"
then
    image_groups=$image_groups' data/light_map'
fi

echo $image_groups

MYCOMMAND='/netshares/scratch/stegmaier/Software/Environments/fusemycells/bin/python /netshares/scratch/stegmaier/Projects/2025/FuseMyCellsISBI_ImageFusion/Source/Source/train_script.py'

MYCOMMAND=$MYCOMMAND' --pipeline imagefusionunet3d --no_dataloader_sanity_check'
MYCOMMAND=$MYCOMMAND' --train_list '$train_list
MYCOMMAND=$MYCOMMAND' --val_list '$val_list
MYCOMMAND=$MYCOMMAND' --test_list '$test_list
MYCOMMAND=$MYCOMMAND' --output_path '$output_path
MYCOMMAND=$MYCOMMAND' --image_groups '$image_groups
MYCOMMAND=$MYCOMMAND' --in_channels '$in_channels
MYCOMMAND=$MYCOMMAND' --log_path '$log_path
MYCOMMAND=$MYCOMMAND' --use_fmc_loss'

echo $MYCOMMAND
eval $MYCOMMAND
