import os
import os.path as osp

root_dir = f"/data/xiaotong/data/cifar100"

def split_func(mode):
    mode_dir = osp.join(root_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    with open(f"./cifar100_{mode}.txt") as f:
        lines = f.readlines()
    for line in lines:
        rel_path = line.strip('\n')
        # 图片源路径在 root_dir 下通常按模式存放（如 train/<class>/<file> 或 val/<class>/<file>）
        # 所以在查找源文件时需要包含 mode 子目录
        src_path = osp.realpath(osp.join(root_dir, mode, rel_path))
        assert osp.exists(src_path)
        class_dir = osp.realpath(osp.join(mode_dir, osp.dirname(rel_path)))
        if not osp.exists(class_dir):
            os.mkdir(class_dir)
        file_name = osp.basename(src_path)
        dst_path = osp.join(class_dir, file_name)
        os.symlink(src_path, dst_path)

split_func('train')
split_func('val')