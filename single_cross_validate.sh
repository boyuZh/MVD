#!/bin/bash
# Python TypeScript Java CS CCPP JavaScript

# # 检查是否有传递参数
# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 <integer> <train_option>"
#     exit 1
# fi

# # 获取第一个参数，并确保它是整数
# integer=$1
# train_option=$2

# if ! [[ "$integer" =~ ^-?[0-9]+$ ]]; then
#     echo "Error: Argument is not an integer. Input 0 or 1."
#     exit 1
# fi

# This script is just for testing purposes.

for seed in 43 6916 25569 8408 72844 8432 9406 6088 25888 76954
do 
    for lan in Python TypeScript Java CCPP JavaScript
    do
      if [ $integer -eq 1 ] && [ $train_option -eq 1 ]; then
        if [ -e "./saved_models/our${lan}/seed${seed}/checkpoint-best-f1/model.bin" ]; then
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


        python linevul_main.py \
          --output_dir=./saved_models/our${lan}/seed${seed} \
          --model_type=roberta \
          --tokenizer_name=microsoft/codebert-base \
          --model_name_or_path=microsoft/codebert-base \
          --do_test \
          --data_file=/hpcfs/users/a1232991/Data/CVESingle/CVEALL.csv \
          --language $new_list \
          --epochs 10 \
          --block_size 512 \
          --train_batch_size 16 \
          --eval_batch_size 16 \
          --learning_rate 2e-5 \
          --max_grad_norm 1.0 \
          --evaluate_during_training \
          --seed ${seed}
    done
done

