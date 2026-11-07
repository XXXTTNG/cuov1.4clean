__all__ = [
    "GlobalVarsManager",
    "get_args",
    "seed_etc_options",
    "set_model_mode",
    "set_learning_rates",
    "train_one_epoch",
    "save_checkpoint",
    "task_ending_info",
    "find_not_pretrained_params",
    "modify_head",
]

#  GlobalVarsManager(全局状态)、参数管理、单 epoch 训练、get_args    

from time import time as ttime
import argparse
import random
from collections import OrderedDict
import tqdm
from typing import Literal
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch import optim
import torchvision
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

import utils.vit_builder
from utils.vit_builder import VisionTransformer
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc
from loguru import logger


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]  # {original_label: (taskid, local_label, global_label)}

    def init_from_args(self, args, this_file=__file__):
        self.args = args
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccTaskIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccTaskPredMat=np.zeros([_nt, _nt]), AccClassIncMat=np.zeros([_nt, _nt]),
                                        AccTaskIncList=np.zeros([_nt := self.cl_mngr.num_tasks]), AccTaskPredList=np.zeros([_nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map


def get_args():
    parser = argparse.ArgumentParser(description='Class-Incremental Learning')
    # Experiment options
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet', 'imagenet_a'))
    parser.add_argument('-dr', '--data_root', type=str, default="[PATH_TO_DATASET_DIR]")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 40, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--only_first_task', action='store_true')
    parser.add_argument('--seed', type=int, default=2025)
    # Model options
    parser.add_argument('-m', '--model', type=str, default='vit_base_patch16_224.augreg_in21k', help='vit_base_patch16_224.augreg_in21k')
    parser.add_argument('--lora_start_block', type=int, default=0, help='')
    parser.add_argument('--lora_end_block', type=int, default=11, help='')
    parser.add_argument('--lora_rank', type=int, default=0, help='')
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    # Data augmentation options
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug'), default='autoaug')
    parser.add_argument('--interp_mode', type=str, choices=('auto', 'bilinear', 'bicubic'), default='auto')
    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=30)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--use_compile', action='store_true')
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('lora', 'head'))
    # Solution Theta options
    parser.add_argument('--use_theta_bl', type=misc.str2bool, default=True)
    parser.add_argument('--use_theta_h', type=misc.str2bool, default=True)
    parser.add_argument('--use_ftinit', type=misc.str2bool, default=True)
    parser.add_argument('--theta_h_alpha', type=float, default=0.3)
    parser.add_argument('--theta_h_beta', type=float, default=200)
    parser.add_argument('--theta_lambda', type=float, default=1e-5)
    parser.add_argument('--theta_device', type=str, default='cuda', choices=('cpu', 'cuda'),
                        help='Device used for streaming CUO statistics and FP64 linear solves')
    parser.add_argument('--theta_dtype', type=str, default='float32', choices=('float32', 'float64'))
    parser.add_argument('--theta_bl_batches', type=int, default=5)
    parser.add_argument('--theta_h_batches', type=int, default=100)
    parser.add_argument('--theta_bl_min_samples', type=int, default=500,
                        help='Minimum effective image samples per task for backbone CUO statistics')
    parser.add_argument('--theta_h_min_samples', type=int, default=0,
                        help='Minimum effective samples per task for head CUO; 0 uses model.embed_dim')
    parser.add_argument('--cuo_infer_mode', type=str, default='cuo', choices=('cuo',),
                        help='CUO inference mode; only the original cumulative cuo mode is supported')
    # Evaluation options
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--only_last_metric', type=misc.str2bool, default=False)
    # Optimizer options
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_scale', type=float, default=1.)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)  # for cosine and warmup
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])  # for multistep
    parser.add_argument('--decay_epochs', type=int, default=1000)  # for step
    parser.add_argument('--decay_rate', type=float, default=0.1)  # for step and multistep
    # Display options
    parser.add_argument('--show_bar', action='store_true')
    parser.add_argument('--print_model', action='store_true')

    args = parser.parse_args()
    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, to_gpu: bool = True, verbose: bool = False, training_string: tuple[str] = ('lora', 'head'), ignore_pretrained_check: bool = False) -> VisionTransformer:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        if training and any([_s == n for _s in training_string]):
            # if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if verbose:
        logger.info(f"parameter.requires_grad ({len(params_requires_grad)}): {params_requires_grad}")
        logger.info(f"module.training: {modules_training}")

    if training:
        if not ignore_pretrained_check:
            for n in GVM.cache_dict['not_pretrained_params']:
                if n not in params_requires_grad:
                    raise ValueError(f"'{n}' does not require grad but it is in the 'not_pretrained_params' list: {GVM.cache_dict['not_pretrained_params']}.")
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model


def set_learning_rates(GVM, model: VisionTransformer, base_lr: float, lr_scale: float, lr_scale_patterns: str, verbose: bool = True) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    if verbose:
        for _lr, _pn_list in lr_param_dict.items():
            logger.info(f'lr={_lr} ({len(_pn_list)}): {_pn_list}')
    return param_lr_groups


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model: VisionTransformer, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer) -> str:
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(lr="step_last:.3e", ce_loss="samp_avg:.4f",
                                    loss="samp_avg:.4f", data_time="step_sum:.3f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%", acc_topk="samp_avg:>6.2%")
    _btimer = ttime()

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        data_time = ttime() - _btimer

        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = model(images)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        ce_loss = criterion(logits / temperature, target)

        loss: Tensor = ce_loss

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1, acc_topk = misc.calc_accuracy(logits, target, topk=(1, 2))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), lr=optimizer.param_groups[0]['lr'], ce_loss=ce_loss.item(),
                                    loss=loss.item(), data_time=data_time, batch_time=batch_time, acc_top1=acc_top1, acc_topk=acc_topk)
        _btimer = ttime()

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def save_checkpoint(GVM: GlobalVarsManager, taskid: int, epoch: int, model: VisionTransformer, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.cpu().clone()
        else:
            if taskid == 0:
                base_params[n] = p.cpu().clone()
            else:
                if not torch.all((_pb := GVM.param_dict[f'base_params'][n]) == p.to(_pb.device)):
                    logger.info(f"WARNING:: save_checkpoint(): 'base_params' is changed!")

    if taskid == 0:
        assert not 'base_params' in GVM.param_dict
        GVM.param_dict['base_params'] = base_params

    GVM.param_dict[f'task_params_{taskid}'] = task_params


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    logger.info(f"{'=' * 90} End of task [{current_taskid + 1}/{GVM.cl_mngr.num_tasks}] {'=' * 90}")
    logger.info("".join([f"\n{_acc_name}(task {current_taskid + 1}):\n{_acc_mat}" for _acc_name, _acc_mat in GVM.acc_mat_dict.items()]))

    acc_info_dict = {
        'diag_task_avg_acc': float(np.diag(GVM.acc_mat_dict['AccTaskIncMat'])[:current_taskid + 1].mean()), # 对角线值（每次任务刚学习完的精度）
        'task_inc_last_acc': float(GVM.acc_mat_dict['AccTaskIncList'][current_taskid]),                     # last_acc（task-id已知）
        'task_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccTaskIncMat'], current_taskid),      # 遗忘率（task-id已知）
        'task_pred_last_acc': float(GVM.acc_mat_dict['AccTaskPredList'][current_taskid]),                   # “预测样本属于哪个任务”的精度，恒为0
        'task_pred_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccTaskPredMat'], current_taskid),    # “预测样本属于哪个任务”的精度，恒为0
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),                   # last_acc（task-id未知）
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),    # 遗忘率（task-id未知）
    }
    _formatter = misc.ScalarFormatter(sep=' | ', diag_task_avg_acc=">6.2%", task_inc_last_acc=">6.2%", task_pred_last_acc=">6.2%", task_pred_last_forg=">6.2%", class_inc_last_acc=">6.2%", task_inc_last_forg=">6.2%", class_inc_last_forg=">6.2%")

    logger.info(f"** Results of task [{current_taskid + 1}]: [ {_formatter(**acc_info_dict)} ] **")
    logger.info(f"** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")


def find_not_pretrained_params(model: VisionTransformer, pretrained: bool = True, pretrained_cfg: dict[str, str] = None, extra_pretrained_params: list[str] = [], verbose: bool = False) -> list[str]:
    assert isinstance(extra_pretrained_params, (list, tuple))

    assert pretrained_cfg is not None

    if 'open_clip' in pretrained_cfg.get('hf_hub_filename', ''):
        _filename = timm.models._hub.HF_OPEN_CLIP_WEIGHTS_NAME
    else:
        _filename = timm.models._hub.HF_WEIGHTS_NAME
    pre_state_dict: OrderedDict[str, Tensor] = timm.models.load_state_dict_from_hf(pretrained_cfg['hf_hub_id'], _filename)

    if 'visual.class_embedding' in pre_state_dict.keys():
        pre_state_dict = timm.models.vision_transformer._convert_openai_clip(pre_state_dict, model)

    not_pretrained_params = []
    for n, p in model.named_parameters():
        if n not in pre_state_dict.keys() or not pretrained:
            not_pretrained_params.append(n)
        else:
            if p.shape != pre_state_dict[n].shape:
                not_pretrained_params.append(n)

    for n in deepcopy(not_pretrained_params):
        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    if verbose:
        logger.info(f"not_pretrained_params({len(not_pretrained_params)})={not_pretrained_params}")
    return not_pretrained_params



def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, verbose: bool = True, ignore_requires_grad_check: bool = False, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if model.head.out_features != len(_target_classes):
        _mh = deepcopy(model.head)
        _mdevice = _mh.weight.device
        _mdtype = _mh.weight.dtype
        model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
        model.head.requires_grad_(_mh.weight.requires_grad)

        if training:
            if not ignore_requires_grad_check:
                assert model.head.weight.requires_grad
        else:
            assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
            _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
            assert model.head.weight.data.shape == _hw.shape
            model.head.weight.data = _hw

            if _mh.bias is not None:
                _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.bias.data.shape == _hb.shape
                model.head.bias.data = _hb
        if verbose:
            logger.info(f" out_features: {_mh.out_features} -> {model.head.out_features}")
    else:
        if verbose:
            logger.info(f"The head does not need to be modified.")

    return model
