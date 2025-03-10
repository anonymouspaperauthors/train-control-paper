# TrainCoCo

The training code in this folder is a wrapper over HuggingFace trainer stack.

## Installation of the trainer stack

```
pip install -e .
```

## Sample TrainCoCo configuration file

Below configuration file is used as part of the evaluation setup which is the common configuration yaml used for all 20 experiments.
```
controller_metrics:
  - name: eval_loss_window_10
    class: HistoryBasedMetric
    arguments:
      window_size: 10
  - name: eval_loss_window_50
    class: HistoryBasedMetric
    arguments:
      window_size: 50
controllers:
  - name: save_when_eval_drop_15
    triggers:
      - on_step_end
    rule: len(eval_loss_window_10["metrics"]["eval_loss"]) > 1 and eval_loss_window_10["metrics"]["eval_loss"][-1] <= 0.85 * eval_loss_window_10["metrics"]["eval_loss"][-2]
    patience:
      patience_threshold: 1
    operations:
      - hfcontrols.should_save
  - name: stop_when_eval_conseq_50_steps_no_change
    triggers:
      - on_step_end
    rule: len(eval_loss_window_50["metrics"]["eval_loss"]) > 49 and eval_loss_window_50["metrics"]["epoch"][-1] > 0.30 and 0.95 * eval_loss_window_50["metrics"]["eval_loss"][-1] <= sum(eval_loss_window_50["metrics"]["eval_loss"])/len(eval_loss_window_50["metrics"]["eval_loss"]) <= 1.05 * eval_loss_window_50["metrics"]["eval_loss"][-1]
    patience:
      patience_threshold: 20
    operations:
      - hfcontrols.should_training_stop
```

## Sample training command for manual approach

```
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1234
export WORLD_SIZE=1
export MODEL=ibm-granite/granite-3.1-2b-base
export OUT_DIR=./train_output
export PER_D_BS=4
export TRAIN_PATH="path to a train dataset file"
export VAL_PATH="path to a validation dataset file"

accelerate launch --num_processes=4   --dynamo_backend="no"   --fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP"   --fsdp_cpu_ram_efficient_loading="true"   --fsdp_forward_prefetch="false"   --fsdp_offload_params="false"   --fsdp_sharding_strategy="FULL_SHARD"   --fsdp_state_dict_type="FULL_STATE_DICT"   --fsdp_sync_module_states="true"   --machine_rank="${RANK}"   --main_process_ip="${MASTER_ADDR}"   --main_process_port="${MASTER_PORT}"   --mixed_precision="no"   --num_machines="${WORLD_SIZE}"   --rdzv_backend="static"   --same_network   --use_fsdp   -m tuning.sft_trainer   --gradient_accumulation_steps="1"   --gradient_checkpointing="true"  --warmup_ratio 0.2 --lr_scheduler_type "cosine" --include_tokens_per_second="true"   --learning_rate="1e-05"   --logging_steps="1"   --logging_strategy="steps"   --max_seq_length="4096"   --num_train_epochs="4"   --model_name_or_path="${MODEL}"   --output_dir="${OUT_DIR}"   --per_device_train_batch_size="${PER_D_BS}"   --save_strategy="epoch" --training_data_path="${TRAIN_PATH}" --validation_data_path "${VAL_PATH}"   --use_flash_attn="true" --eval_strategy steps --eval_steps 1 2>&1 | tee -a "${OUT_DIR}/stdout.log"
```

## Sample training command for TrainCoCo approach

You would need to set `traincoco_config_file` to the control configuration yaml file.

```
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1234
export WORLD_SIZE=1
export MODEL=ibm-granite/granite-3.1-2b-base
export OUT_DIR=./train_output
export PER_D_BS=4
export TRAIN_PATH="path to a train dataset file"
export VAL_PATH="path to a validation dataset file"

accelerate launch --num_processes=4   --dynamo_backend="no"   --fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP"   --fsdp_cpu_ram_efficient_loading="true"   --fsdp_forward_prefetch="false"   --fsdp_offload_params="false"   --fsdp_sharding_strategy="FULL_SHARD"   --fsdp_state_dict_type="FULL_STATE_DICT"   --fsdp_sync_module_states="true"   --machine_rank="${RANK}"   --main_process_ip="${MASTER_ADDR}"   --main_process_port="${MASTER_PORT}"   --mixed_precision="no"   --num_machines="${WORLD_SIZE}"   --rdzv_backend="static"   --same_network   --use_fsdp   -m tuning.sft_trainer   --gradient_accumulation_steps="1"   --gradient_checkpointing="true"  --warmup_ratio 0.2 --lr_scheduler_type "cosine" --include_tokens_per_second="true"   --learning_rate="1e-05"   --logging_steps="1"   --logging_strategy="steps"   --max_seq_length="4096"   --num_train_epochs="4"   --model_name_or_path="${MODEL}"   --output_dir="${OUT_DIR}"   --per_device_train_batch_size="${PER_D_BS}" --training_data_path="${TRAIN_PATH}" --validation_data_path "${VAL_PATH}"   --use_flash_attn="true" --eval_strategy steps --eval_steps --traincoco_config_file ./tc.yaml 2>&1 | tee -a "${OUT_DIR}/stdout.log"
```
