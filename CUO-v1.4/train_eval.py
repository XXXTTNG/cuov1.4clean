import sys
import os
os.environ['TIMM_FUSED_ATTN'] = '1' if 'TIMM_FUSED_ATTN' not in os.environ else os.environ['TIMM_FUSED_ATTN']
from time import time as ttime
from collections import OrderedDict
import tqdm
from copy import deepcopy
import requests
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from utils.vit_builder import VisionTransformer
from utils.dataset_builder import define_dataset
from utils import misc
from loguru import logger
from utils.train_funcs import *
from utils.interm_tensor_funcs import *

torch.set_float32_matmul_precision("high")


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: VisionTransformer, **kwargs) -> VisionTransformer:
    args = GVM.args
    if args.epochs == 0:
        return model

    logger.info(f"*" * 90 + " Start Training " + "*" * 90)

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(GVM, model, training=True, verbose=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    if args.lora_rank > 0 and taskid > 0 and args.use_theta_bl and args.use_ftinit:
        logger.info(f"Initializing LoRA parameters for task {taskid} using Forward Transfer Initialization ...")
        lora_param_names = [name for name in model.state_dict().keys() if 'lora_' in name]

        if lora_param_names:
            prev_param_means = {}
            for param_name in lora_param_names:
                param_values = []
                for prev_taskid in range(taskid):
                    if f'task_params_{prev_taskid}' in GVM.param_dict:
                        prev_params = GVM.param_dict[f'task_params_{prev_taskid}']
                        if param_name in prev_params:
                            param_values.append(prev_params[param_name])
                if param_values:
                    mean_param = torch.stack(param_values).mean(dim=0)
                    prev_param_means[param_name] = mean_param

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in prev_param_means:
                        param.copy_(prev_param_means[name])
                        logger.info(f"Initialized {name} with mean of {len([k for k in GVM.param_dict.keys() if k.startswith('task_params_') and int(k.split('_')[-1]) < taskid])} previous tasks")

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times, verbose=True)
    logger.info(f"{str(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns, verbose=True)

    optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer)
            logger.info(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)

    save_checkpoint(GVM, taskid, epoch, model, optimizer, scheduler)

    idevice: str = args.theta_device
    idtype = {'float32': torch.float32, 'float64': torch.float64}.get(args.theta_dtype)

    if args.use_theta_bl:
        logger.info("-" * 90 + f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Using Cumulative Unified Optimization for backbone " + "-" * 90)
        if taskid == 0:
            GVM.cache_dict['param_id_dict'] = get_param_id_dict(GVM, model, ['qkv.weight'])

        _interm_tensor_dict = get_interm_tensor_dict(GVM, model, GVM.cache_dict['param_id_dict'])
        calc_and_store_solution_components(
            GVM, _interm_tensor_dict, GVM.cache_dict['param_id_dict'],
            idevice=idevice, idtype=idtype,
        )
        _Theta_bl_t_dict = calc_solution_Theta_bl_t(GVM, eps=args.theta_lambda, idevice=idevice, idtype=idtype)
        _Theta_bl_t_dict = convert_Theta_bl_t_dict_keys(GVM, model, _Theta_bl_t_dict)
        GVM.cache_dict['Theta_bl_t_dict'] = _Theta_bl_t_dict
        model = distribute_solution_Theta_bl_t(GVM, model, _Theta_bl_t_dict)
        del _interm_tensor_dict

    if args.use_theta_h:
        logger.info("-" * 90 + f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Using Cumulative Unified Optimization for Head " + "-" * 90)
        if args.use_theta_bl:
            set_model_flag(model, 'use_Theta_bl_t_for_infer', True, verbose=True)

        _head_interm_tensor = get_head_interm_tensors(GVM, model, idevice=idevice, idtype=idtype, verbose=True)
        calc_and_store_head_solution_components(GVM, _head_interm_tensor, idevice=idevice, idtype=idtype)
        _Theta_h_t = calc_solution_Theta_h_t(GVM, eps=args.theta_lambda, idevice=idevice, idtype=idtype)
        model = distribute_solution_Theta_h_t(GVM, model, _Theta_h_t)

        del _head_interm_tensor, _Theta_h_t

        if args.use_theta_bl:
            set_model_flag(model, 'use_Theta_bl_t_for_infer', False, verbose=True)

    logger.info(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False, verbose=train_taskid == 0)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    scalar_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%")

    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = model(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        class_inc_preds = logits.argmax(dim=1)
        _task_inc_logits = logits.clone()
        _task_inc_logits[:, :eval_taskid * GVM.cl_mngr.num_classes_per_task] = -torch.inf
        _task_inc_logits[:, (eval_taskid + 1) * GVM.cl_mngr.num_classes_per_task:] = -torch.inf
        task_inc_preds = _task_inc_logits.argmax(dim=1)

        acc_task_inc, _, _ = misc.calc_acc_topnn_dynamically(task_inc_preds, target)
        acc_class_inc, _, _ = misc.calc_acc_topnn_dynamically(class_inc_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_task_inc=acc_task_inc, acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    logger.info(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)
    result_dict['task_pred_acc'] = 0.

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: VisionTransformer):
    logger.info(f"*" * 90 + f" Start Evaluation " + "*" * 90)
    args = GVM.args

    model = modify_head(GVM, model, training=False)

    if args.use_theta_bl and train_taskid > 0:
        set_model_flag(model, 'use_Theta_bl_t_for_infer', True, verbose=True)
    if args.use_theta_h and train_taskid > 0:
        set_model_flag(model, 'use_Theta_h_t_for_infer', True, verbose=True)

    average_acc_meter = misc.ScalarMeter(acc_task_inc="samp_avg:>6.2%", acc_class_inc="samp_avg:>6.2%", task_pred_acc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccTaskIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_task_inc']
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        GVM.acc_mat_dict[f'AccTaskPredMat'][train_taskid, eval_taskid] = one_result_dict['task_pred_acc']
        average_acc_meter.add_step_value(**one_result_dict)

    if args.use_theta_bl and train_taskid > 0:
        set_model_flag(model, 'use_Theta_bl_t_for_infer', False, verbose=True)
    if args.use_theta_h and train_taskid > 0:
        set_model_flag(model, 'use_Theta_h_t_for_infer', False, verbose=True)

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccTaskIncList'][train_taskid] = avg_result_dict['acc_task_inc']
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']
    GVM.acc_mat_dict[f'AccTaskPredList'][train_taskid] = avg_result_dict['task_pred_acc']

@logger.catch()
def main_func():
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args, this_file=__file__)

    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runlogs')
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args.dataset}_t{args.num_tasks}_cuo_seed{args.seed}_{{time:YYYYMMDD_HHmmss}}.log")
    misc.set_logger('main', log_path=_log_path)
    logger.info(misc.format_args(args))
    logger.info(f"Command: \"python {' '.join(sys.argv)}\"")
    GVM.cache_dict['exp_start_time'] = ttime()
    logger.info(f"Class sequence: {GVM.cl_mngr.task_class_list}")

    for taskid, current_task_classes in GVM.cl_mngr:
        logger.info(f"{'#' * 90} Task: [{taskid + 1}/{GVM.cl_mngr.num_tasks}] {'#' * 90}")
        logger.info(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if taskid == 0:
            _lora_args_dict = misc.get_specific_args_dict(args, 'lora_')

            logger.info(f"Creating model ({args.model}) using timm ...")
            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, lora_args_dict=_lora_args_dict)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)

            logger.info(f"Created a new model: {model.__class__.__name__}")

        if args.use_compile:
            # model = torch.compile(model)
            raise NotImplementedError("compile() is not supported in this version.")

        if taskid == 0:
            logger.info(f"model.arch ({args.model})" + (f":\n{str(model)}" if args.print_model else ""))
            logger.info(f"model.pretrained_cfg={GVM.cache_dict['pretrained_cfg']}")

        # task 0 必须成功建立 not_pretrained_params 基线(SSL 等异常照常抛出); 后续任务容忍 SSL 错误并沿用旧值
        try:
            GVM.cache_dict['not_pretrained_params'] = find_not_pretrained_params(model, pretrained_cfg=model.pretrained_cfg, verbose=False)
            logger.info(f"not_pretrained_params({len(GVM.cache_dict['not_pretrained_params'])})={GVM.cache_dict['not_pretrained_params']}")
        except requests.exceptions.SSLError as e:
            if taskid == 0:
                raise
            logger.info(f"SSL Error occurred while checking pretrained parameters: {str(e)}")
            logger.info("'not_pretrained_params' does not change.")

        # Training
        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict['training_string'] = args.training_string
        misc.check_param_training(GVM.cache_dict['not_pretrained_params'], GVM.cache_dict['training_string'])

        model = train_one_task(GVM, taskid, current_task_classes, model)

        # Evaluation
        if args.only_last_metric:
            if taskid + 1 < GVM.cl_mngr.num_tasks:
                logger.info("[Eval] only_last_metric=True, skip intermediate evaluation.")
                continue

        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)

        if args.only_first_task:
            break


if __name__ == "__main__":
    main_func()
