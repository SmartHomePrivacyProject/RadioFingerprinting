#! /bin/bash

echo 'setup variable values'

if [ $? -eq 0 ]
then
    echo 'nvidia-smi check pass' `date`
    echo 'will run with GPU'
    export useGpu=1
else
    echo 'nvidia-smi not exists, will run with CPU'
    export useGpu=-1
fi

SHELL_FOLDER=$(dirname $(readlink -f "$0"))
export ROOT_DIR=$SHELL_FOLDER
echo ROOT_DIR=$ROOT_DIR

export ResRootDir=$ROOT_DIR/res_out_root
echo ResRootDir=$ResRootDir

export PYTHONPATH=$PYTHONPATH:/home/haipeng/Documents/fine-tuning

echo 'start running...'
python3 $*
