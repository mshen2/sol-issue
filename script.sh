export CUDA_VISIBLE_DEVICES=0


############### deepspeed

# not work either llama or openllama
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path openlm-research/open_llama_3b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./out/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./default_offload_opt_param.json"




############### fsdp

# openllama works
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path openlm-research/open_llama_3b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./out/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'

# change to llama2 does not work
torchrun --nproc_per_node=2 train.py \
    --model_name_or_path /scratch/mshen16/llama2/7b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./out/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'





    



export HF_HOME="/scratch/mshen16/.cache"
CUDA_VISIBLE_DEVICES=0,1 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate


accelerate launch examples/pytorch/summarization/run_summarization_no_trainer.py \
    --model_name_or_path t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./tmp/tst-summarization \
    --per_device_train_batch_size=4


