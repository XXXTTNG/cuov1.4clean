python train_eval.py --training_string lora head --lora_rank 8 --lr 0.0001 -b 256 --lr_scale_patterns head --lr_scale 100 --theta_device cuda $@
