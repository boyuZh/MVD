for lan in Python TypeScript Java CS CCPP JavaScript
    do
    python linevul_main.py \
  --output_dir=./saved_models/our${lan} \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=/hpcfs/users/a1232991/Data/CVESingle/CVE_${lan}/CVE${lan}_train.csv \
  --eval_data_file=/hpcfs/users/a1232991/Data/CVESingle/CVE_${lan}/CVE${lan}_val.csv \
  --test_data_file=/hpcfs/users/a1232991/Data/CVESingle/CVE_${lan}/CVE${lan}_test.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 43
done