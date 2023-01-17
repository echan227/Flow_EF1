#!/bin/bash
echo "DIR": $PWD;

# if [ "$PWD" = "/home/bram/Scripts/AI_CMR_QC" ];
# then
#     export nnUNet_raw_data_base="/home/bram/Models/nnUNet_data/nnUNet_raw_data_base/"
#     export nnUNet_preprocessed="/home/bram/Models/nnUNet_data/nnUNet_preprocessed/"
#     export RESULTS_FOLDER="/home/bram/Models/nnUNet_data/nnUNet_trained_models/"
# elif [ "$PWD" = "/home/br14/code/Python/AI_CMR_QC" ];
# then
export nnUNet_raw_data_base="/data/nnUNet/nnUNet_raw_data_base"
export nnUNet_preprocessed="/data/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/nnUNet/nnUNet_trained_models"



# fi
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
fold0_path="$inferenceDir/fold0/"
fold1_path="$inferenceDir/fold1/"
fold2_path="$inferenceDir/fold2/"
fold3_path="$inferenceDir/fold3/"
fold4_path="$inferenceDir/fold4/"
ensemble_path="$inferenceDir/ensemble/"


nnUNet_predict -i $imagesTs -o $fold0_path -f 0 -t $model_task_id -m 2d --save_npz
nnUNet_predict -i $imagesTs -o $fold1_path -f 1 -t $model_task_id -m 2d --save_npz
nnUNet_predict -i $imagesTs -o $fold2_path -f 2 -t $model_task_id -m 2d --save_npz
nnUNet_predict -i $imagesTs -o $fold3_path -f 3 -t $model_task_id -m 2d --save_npz
nnUNet_predict -i $imagesTs -o $fold4_path -f 4 -t $model_task_id -m 2d --save_npz

nnUNet_ensemble -f $fold0_path $fold1_path $fold2_path $fold3_path $fold4_path -o $ensemble_path
