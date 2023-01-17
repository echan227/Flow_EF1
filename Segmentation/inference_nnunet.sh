#!/bin/bash
echo "DIR": $PWD;

if [ "$PWD" = "/home/bram/Scripts/AI_CMR_QC" ];
then
    export nnUNet_raw_data_base="/home/bram/Models/nnUNet_data/nnUNet_raw_data_base/"
    export nnUNet_preprocessed="/home/bram/Models/nnUNet_data/nnUNet_preprocessed/"
    export RESULTS_FOLDER="/home/bram/Models/nnUNet_data/nnUNet_trained_models/"
elif [ "$PWD" = "/home/br14/code/Python/AI_CMR_QC" ];
then
    export nnUNet_raw_data_base="/data/nnUNet/nnUNet_raw_data_base"
    export nnUNet_preprocessed="/data/nnUNet/nnUNet_preprocessed"
    export RESULTS_FOLDER="/data/nnUNet/nnUNet_trained_models"



fi
export CUDA_VISIBLE_DEVICES=0
export NPY_MKL_FORCE_INTEL=1

while getopts a:b:c: flag
do
    case "${flag}" in
        a) imagesTs=${OPTARG};;
        b) inferenceDir=${OPTARG};;
        c) model_task_id=${OPTARG};;

    esac
done
echo "imagesTs: $imagesTs";
echo "inferenceDir: $inferenceDir";
echo "model_task_id: $model_task_id";


model_trainer_name="nnUNetTrainerV2"
model_planner_name="nnUNetPlansv2.1"
folds="0"

nnUNet_predict -i $imagesTs -o $inferenceDir -t $model_task_id -m 2d -tr $model_trainer_name -p $model_planner_name -f $folds
