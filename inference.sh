#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export NPY_MKL_FORCE_INTEL=1

while getopts a:b:c: flag
do
    case "${flag}" in
        a) imagesTs=${OPTARG};;
        b) inferenceDir=${OPTARG};;

    esac
done
echo "imagesTs: $imagesTs";
echo "inferenceDir: $inferenceDir";


model_trainer_name="nnUNetTrainerV2"
model_planner_name="nnUNetPlansv2.1"
folds="0"
model_task_id = "200"

printf "\n"
echo "========================================================================================="
echo "Running inference for Task$model_task_id"
echo "========================================================================================="
printf "\n"

nnUNet_predict -i $imagesTs -o $inferenceDir -t $model_task_id -m 2d -tr $model_trainer_name -p $model_planner_name -f $folds


# TO RUN IT
# inference.sh -a path_INPUT_DIR -b path_OUTPUT_DIR
