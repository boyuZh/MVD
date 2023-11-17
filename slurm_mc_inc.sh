#!/bin/bash

# 检查是否有传递参数
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <integer> <train_option>"
    exit 1
fi

# 获取第一个参数，并确保它是整数
integer=$1
train_option=$2

if ! [[ "$integer" =~ ^-?[0-9]+$ ]]; then
    echo "Error: Argument is not an integer. Input 0 or 1."
    exit 1
fi

# Conditionally set the --do_train option
if [ $train_option -eq 1 ]; then
    DO_TRAIN="--do_train"
else
    DO_TRAIN=""
fi

for lan in Python TypeScript Java CS CCPP JavaScript
do 
  if [ $integer -eq 1 ] && [ $train_option -eq 1 ]; then
    if [ -e "./saved_models/ourMCInc/Lack_${lan}/checkpoint-best-f1/model.bin" ]; then
        echo "File exists, skipping iteration."
        continue
    fi
  fi
  teacher_lan_list=Python,TypeScript,Java,CS,CCPP,JavaScript
  element_to_remove=${lan}
  # 使用sed命令删除指定元素
  teacher_lan_list=$(echo "$teacher_lan_list" | sed "s/,$element_to_remove,/,/g")
  teacher_lan_list=$(echo "$teacher_lan_list" | sed "s/^$element_to_remove,//")
  new_list=$(echo "$teacher_lan_list" | sed "s/,$element_to_remove$//")
  new_full_list=${new_list},${lan}

  echo $element_to_remove
  echo $new_list
  echo $new_full_list
  
  sbatch <<EOL
#!/bin/bash
#SBATCH -p a100
#SBATCH --nodes 1
#SBATCH -c 16
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --err="_logs_linevul/bigvul_results_mc_inc_${lan}.err"
#SBATCH --output="_logs_linevul/bigvul_results_mc_inc_${lan}.out"
#SBATCH --job-name="Big_Linevul"

## Setup Python Environment
module purge
module use /apps/skl/modules/all
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.6.2
module load NCCL/2.12.12-GCCcore-11.2.0-CUDA-11.6.2
export HF_HOME=/hpcfs/users/a1232991/.cache/huggingface/
# activate conda env

source /hpcfs/users/a1232991/local/virtualenvs/llm/bin/activate
deactivate
source /hpcfs/users/a1232991/local/virtualenvs/llm/bin/activate




python linevul_main_mc.py \
  --output_dir=./saved_models/ourMCIncTeacher/Lack_${lan} \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  $DO_TRAIN \
  --do_test \
  --data_file=/hpcfs/users/a1232991/Data/CVESingle/CVEALL.csv \
  --language $new_list \
  --epochs 10 \
  --block_size 512 \
  --focal-loss \
  --logits-adjust \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 43

python linevul_main_mc_inc.py \
  --output_dir=./saved_models/ourMCInc/Lack_${lan} \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  $DO_TRAIN \
  --do_test \
  --data_file=/hpcfs/users/a1232991/Data/CVESingle/CVEALL.csv \
  --language $new_full_list \
  --n_old_class 5 \
  --resume ./saved_models/ourMCIncTeacher/Lack_${lan}/checkpoint-best-f1/model.bin \
  --epochs 10 \
  --focal-loss \
  --logits-adjust \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 43
EOL
done