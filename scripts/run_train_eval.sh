# parameters
model=simclr
epochs=800
augs=default
num_runs=1
timestamp=`date '+%s'`

# preparation
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd ${SHELL_FOLDER}/../
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# train and eval
python train_eval.py --model=${model} --epochs=${epochs} --augs=${augs} --num_runs=${num_runs} | tee ./results/"${timestamp}_${model}_epochs_${epochs}_augs_${augs}".txt
