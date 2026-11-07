CUDA_VISIBLE_DEVICES=0 bash script_ir.sh   -t 10 --data_root "/data/xiaotong/dataset/imagenet-r"
CUDA_VISIBLE_DEVICES=3 bash script_ir.sh   -t 20 --data_root "/data/xiaotong/dataset/imagenet-r"
CUDA_VISIBLE_DEVICES=0 bash script_c100.sh -t 10 --data_root "/data/xiaotong/data/cifar100"
CUDA_VISIBLE_DEVICES=1 bash script_idn.sh  -t 10 --data_root "/data/xiaotong/data/DomainNet"
CUDA_VISIBLE_DEVICES=2 bash script_ia.sh   -t 20 --data_root "/data/xiaotong/data/imageNet-A"




# cuo CUO
CUDA_VISIBLE_DEVICES=0 bash script_ir.sh -t 50 --data_root /data/xiaotong/dataset/imagenet-r --cuo_infer_mode cuo
CUDA_VISIBLE_DEVICES=1 bash script_ir.sh -t 20 --data_root /data/xiaotong/dataset/imagenet-r --cuo_infer_mode cuo
CUDA_VISIBLE_DEVICES=2 bash script_ir.sh -t 10 --data_root /data/xiaotong/dataset/imagenet-r --cuo_infer_mode cuo

CUDA_VISIBLE_DEVICES=0 bash script_idn.sh -t 50 --data_root /data/xiaotong/dataset/domainnet --cuo_infer_mode cuo
CUDA_VISIBLE_DEVICES=1 bash script_idn.sh -t 20 --data_root /data/xiaotong/dataset/domainnet --cuo_infer_mode cuo
CUDA_VISIBLE_DEVICES=2 bash script_idn.sh -t 10 --data_root /data/xiaotong/dataset/domainnet --cuo_infer_mode cuo



