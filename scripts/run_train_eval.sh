# parameters
model=simclr
epochs=800
augs=default
num_runs=1
timestamp=`date '+%s'`

# preparation
cd ../
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# train and eval
python train_eval.py --model=${model} --epochs=${epochs} --augs=${augs} --num_runs=${num_runs} | tee ./results/"${timestamp}_${model}_epochs_${epochs}_augs_${augs}".txt