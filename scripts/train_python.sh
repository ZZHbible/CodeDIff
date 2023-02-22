nohup python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=12233 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 50000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--microbatch 128 \
--dataset code2summary_python \
--data_dir path/to/dataset \
--vocab bert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes test-code2summary_python > log.txt 2>&1 &
