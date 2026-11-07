__all__ = [
    'get_param_id_dict',
    'get_interm_tensor_dict',
    'calc_and_store_solution_components',
    'calc_solution_Theta_bl_t',
    'convert_Theta_bl_t_dict_keys',
    'distribute_solution_Theta_bl_t',
    'set_model_flag',
    'get_head_interm_tensors',
    'calc_and_store_head_solution_components',
    'calc_solution_Theta_h_t',
    'distribute_solution_Theta_h_t',
]

# CUO 算法核心:抓 X/Y/W₀ → 算 F/G → 解 Θ → 写回模型    

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.hooks
from einops import rearrange
from utils.train_funcs import GlobalVarsManager, set_model_mode
from utils.vit_builder import VisionTransformer, IntermReader
from utils.dataset_builder import define_statistics_dataset
from typing import Literal
from collections import OrderedDict
from loguru import logger
from time import time as ttime


def get_param_id_dict(GVM: GlobalVarsManager, model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    logger.info(f"Get param id dict with patterns: {patterns}")
    _func_start = ttime()
    if isinstance(patterns, str):
        patterns = [patterns]
    assert isinstance(patterns, list), f"{patterns} is not a list"
    param_id_dict = {}
    for n, p in model.named_parameters():
        if any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"

    _valid_block_idxs = list(range(GVM.args.lora_start_block, GVM.args.lora_end_block + 1))
    for _pid in list(param_id_dict.keys()):
        _info_dict = param_id_dict[_pid]
        if _info_dict['name'].startswith('blocks.'):
            _block_idx = int(_info_dict['name'].split('.')[1])
            if _block_idx not in _valid_block_idxs:
                _ret = param_id_dict.pop(_pid)
                assert _ret == _info_dict
                continue
    assert len(param_id_dict) == len(_valid_block_idxs), f"{len(param_id_dict)} != {len(_valid_block_idxs)}"

    logger.info(f"Get param id dict with patterns done, time: {ttime() - _func_start:.2f}s")
    return param_id_dict

# 这个函数是抓取x、y以及w0
def get_interm_tensor_dict(GVM: GlobalVarsManager, model: VisionTransformer, dst_param_id_dict: dict[int, dict[Literal['name', 'shape'], str | list[int]]]) -> dict[int, dict[str, Tensor]]:
    logger.info(f"Get interm tensor dict with dst_param_id_dict: {dst_param_id_dict}")
    _func_start = ttime()

    # Only one batch of X/Y is kept alive.  The sufficient statistics are
    # accumulated where they are computed, instead of retaining all tokens from
    # all images (about 13.5 GiB for the default ViT-B/16 configuration).
    interm_tensor_dict: dict[int, dict[str, Tensor]] = OrderedDict()
    pending_tensor_dict: dict[int, dict[str, Tensor]] = {}
    DEVICE = torch.device(GVM.args.theta_device)
    idtype = {'float32': torch.float32, 'float64': torch.float64}[GVM.args.theta_dtype]
    logger.info(f"Streaming backbone CUO statistics on {DEVICE} with dtype={idtype}")
    num_tokens = model.embed_len
    embed_dim = model.embed_dim

    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor):
        if isinstance(module, IntermReader):
            _pid = module.dst_param_id
            _mname = module.module_name
            _interm_tensor: Tensor = args[0]

            if _mname in ('interm_reader_xi'):
                # [B, N, D]
                assert _interm_tensor.shape[1] == num_tokens and _interm_tensor.shape[2] == dst_param_id_dict[_pid]['shape'][-1], f"{_interm_tensor.shape} ; {dst_param_id_dict[_pid]['shape']}"
                assert dst_param_id_dict[_pid]['shape'][-1] == embed_dim, f"{_interm_tensor.shape}"
                _interm_tensor = rearrange(_interm_tensor, 'b n d -> (b n) d')  # [B*N, D]'
            elif _mname == 'interm_reader_yi':
                # [B, N, C=3D]
                assert _interm_tensor.shape[1] == num_tokens and _interm_tensor.shape[2] == dst_param_id_dict[_pid]['shape'][0], f"{_interm_tensor.shape} ; {dst_param_id_dict[_pid]['shape']}"
                assert dst_param_id_dict[_pid]['shape'][0] == embed_dim * 3, f"{_interm_tensor.shape}"
                _interm_tensor = rearrange(_interm_tensor, 'b n c -> (b n) c')  # [B*N, C=3D]
            elif _mname == 'interm_reader_w':
                assert list(_interm_tensor.shape) == dst_param_id_dict[_pid]['shape'], f"{_interm_tensor.shape} ; {dst_param_id_dict[_pid]['shape']}"
                _interm_tensor = rearrange(_interm_tensor, 'c d -> d c')  # [D, C=3D]
            else:
                raise NotImplementedError(_mname)

            if _mname in ('interm_reader_xi', 'interm_reader_yi'):
                pending_tensor_dict.setdefault(_pid, {})[_mname] = _interm_tensor.to(device=DEVICE, dtype=idtype)
            elif _mname == 'interm_reader_w':
                W0 = _interm_tensor.to(device=DEVICE, dtype=idtype)
                pending = pending_tensor_dict[_pid]
                Xi = pending.pop('interm_reader_xi')
                Yi = pending.pop('interm_reader_yi')
                assert not pending, f"Unconsumed hook tensors for pid={_pid}: {pending.keys()}"

                Fi = Xi.T @ Xi
                Gi = Xi.T @ (Yi - Xi @ W0)
                if _pid not in interm_tensor_dict:
                    interm_tensor_dict[_pid] = {'Fi': Fi, 'Gi': Gi, 'W0': W0}
                else:
                    interm_tensor_dict[_pid]['Fi'].add_(Fi)
                    interm_tensor_dict[_pid]['Gi'].add_(Gi)
                    assert torch.equal(interm_tensor_dict[_pid]['W0'], W0)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    _handle_list: list[torch.utils.hooks.RemovableHandle] = []
    for n, m in model.named_modules():
        if 'interm_reader' in n and isinstance(m, IntermReader):
            _handle_list.append(m.register_forward_hook(_forward_hook))

    model = set_model_mode(GVM, model, training=False)
    torch.cuda.empty_cache()

    args = GVM.args
    max_samples = args.eval_batch_size * args.theta_bl_batches
    assert max_samples >= args.theta_bl_min_samples, \
        f"theta_bl capacity {max_samples} is smaller than theta_bl_min_samples={args.theta_bl_min_samples}"
    dataset, sample_info = define_statistics_dataset(
        GVM, GVM.cl_mngr.current_task_classes, transform_type=args.transform_type,
        min_samples=args.theta_bl_min_samples, max_samples=max_samples,
        target_map_to_local=args.seperate_head,
        seed=args.seed + GVM.cl_mngr.current_taskid * 1009 + 11)
    assert len(dataset) >= args.theta_bl_min_samples
    logger.info(f"[CUO Sampling][Backbone][Task {GVM.cl_mngr.current_taskid + 1}] {sample_info}")
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    for img, _ in dataloader:
        with torch.no_grad():
            img: Tensor
            model(img.cuda(non_blocking=True))
    torch.cuda.empty_cache()

    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0
    assert not any(pending_tensor_dict.values()), f"Unconsumed hook tensors: {pending_tensor_dict.keys()}"
    assert sorted(list(interm_tensor_dict.keys())) == sorted(list(dst_param_id_dict.keys())), f"!! ({len(interm_tensor_dict.keys())}){interm_tensor_dict.keys()} \n != ({len(dst_param_id_dict.keys())}) {dst_param_id_dict}"

    for _pid, _mname_dict in interm_tensor_dict.items():
        assert sorted(_mname_dict) == ['Fi', 'Gi', 'W0']
        assert _mname_dict['Fi'].shape == (embed_dim, embed_dim)
        assert _mname_dict['Gi'].shape == _mname_dict['W0'].shape
        assert _mname_dict['W0'].T.shape == tuple(dst_param_id_dict[_pid]['shape'])

    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataset))

    logger.info(f"Get interm tensor dict done ({len(dataset)} samples), time: {ttime() - _func_start:.2f}s")
    return interm_tensor_dict


# 这个函数是计算三个量得到F、G以及W0，并存储在GVM.cache_dict['Theta_bl_t-cums']中，供后续计算Theta_bl_t使用
def calc_and_store_solution_components(GVM: GlobalVarsManager, interm_tensor_dict: dict[int, dict[str, Tensor]], dst_param_id_dict: dict[int, dict[Literal['name', 'shape'], str | list[int]]], idevice: str = 'cpu', idtype=torch.float32):
    logger.info(f"Calc and store solution components")
    _func_start = ttime()
    DEVICE = torch.device(idevice)

    solu_comp_dict_i: dict[int, dict[str, Tensor]] = OrderedDict()
    for _pid, _mname_dict in interm_tensor_dict.items():
        if sorted(list(_mname_dict.keys())) == ['Fi', 'Gi', 'W0']:
            Fi = _mname_dict['Fi'].to(device=DEVICE, dtype=idtype)
            Gi = _mname_dict['Gi'].to(device=DEVICE, dtype=idtype)
            W0 = _mname_dict['W0'].to(device=DEVICE, dtype=idtype)
            assert Fi.shape[0] == Fi.shape[1] == W0.shape[0], f"{Fi.shape}; {W0.shape}"
            assert Gi.shape[0] == W0.shape[0] and Gi.shape[1] == W0.shape[1], f"{Gi.shape}; {W0.shape}"

            solu_comp_dict_i[_pid] = {'Fi': Fi.cpu().float(), 'Gi': Gi.cpu().float(), 'W0': W0.cpu().float()}
        else:
            raise NotImplementedError(f"Unexpected interm tensor dict keys: {_mname_dict.keys()}")
    assert len(solu_comp_dict_i) == (_l :=  (GVM.args.lora_end_block + 1 - GVM.args.lora_start_block)), f"{len(solu_comp_dict_i)} != {_l}; {len(interm_tensor_dict)}"

    if 'Theta_bl_t-cums' not in GVM.cache_dict:
        GVM.cache_dict['Theta_bl_t-cums'] = OrderedDict()
        for _pid, _str_tensor_dict in solu_comp_dict_i.items():
            GVM.cache_dict['Theta_bl_t-cums'][_pid] = {}
            GVM.cache_dict['Theta_bl_t-cums'][_pid]['Ft'] = _str_tensor_dict['Fi'].cpu().float()
            GVM.cache_dict['Theta_bl_t-cums'][_pid]['Gt'] = _str_tensor_dict['Gi'].cpu().float()
            GVM.cache_dict['Theta_bl_t-cums'][_pid]['W0'] = _str_tensor_dict['W0'].cpu().float()
    else:
        assert GVM.cl_mngr.current_taskid > 0, f"{GVM.cl_mngr.current_taskid}"
        for _pid, _str_tensor_dict in solu_comp_dict_i.items():
            GVM.cache_dict['Theta_bl_t-cums'][_pid]['Ft'] += _str_tensor_dict['Fi'].cpu().float()
            GVM.cache_dict['Theta_bl_t-cums'][_pid]['Gt'] += _str_tensor_dict['Gi'].cpu().float()
            assert torch.all(GVM.cache_dict['Theta_bl_t-cums'][_pid]['W0'] == _str_tensor_dict['W0'])

    for _pid, _mname_dict in list(interm_tensor_dict.items()):
        for _mname in ['Fi', 'Gi', 'W0']:
            if _mname in _mname_dict:
                del interm_tensor_dict[_pid][_mname]
    logger.info(f"Calc and store solution components done, time: {ttime() - _func_start:.2f}s")


def calc_solution_Theta_bl_t(GVM: GlobalVarsManager, eps: float = 1e-5, idevice: str = 'cpu', idtype=torch.float32) -> OrderedDict[int, Tensor]:
    logger.info(f"Calc solution Theta_bl_t")
    _func_start = ttime()

    if 'Theta_bl_t-cums' not in GVM.cache_dict:
        raise NotImplementedError("!!")
    solu_comp_dict: dict[int, dict[str, Tensor]] = GVM.cache_dict['Theta_bl_t-cums']

    Theta_bl_t_dict: OrderedDict[int, Tensor] = OrderedDict()
    for _pid, _str_tensor_dict in solu_comp_dict.items():
        # Accumulators stay in CPU/FP32 between tasks; solve on the requested
        # device in FP64 for numerical stability.
        Ft: Tensor = _str_tensor_dict['Ft'].to(device=idevice, dtype=torch.float64)  # [D, D]
        Gt: Tensor = _str_tensor_dict['Gt'].to(device=idevice, dtype=torch.float64)  # [D, C]
        Ft.diagonal().add_(eps)
        Theta_bl_t: Tensor = torch.linalg.solve(Ft, Gt)  # [D, D] × [D, C] -> [D, C]
        assert Theta_bl_t.shape == _str_tensor_dict['W0'].shape, f"{Theta_bl_t.shape} != {_str_tensor_dict['W0'].shape}"
        Theta_bl_t_dict[_pid] = Theta_bl_t.cpu().float()  # model buffers are FP32

    logger.info(f"Calc solution Theta_bl_t done, time: {ttime() - _func_start:.2f}s")
    return Theta_bl_t_dict


def _find_buffer_with_pid(_pid: int, _name, model: VisionTransformer, level: int = 2, verbose: bool = False) -> str:
    for n, p in model.named_parameters():
        if id(p) == _pid:
            assert '.' in n
            parent_module_name = n.rsplit('.', level)[0]
            if verbose:
                logger.info(f"parent_module_name: {parent_module_name}")
            for buf_name, _ in model.named_buffers():
                if buf_name == parent_module_name + '.' + _name:
                    if verbose:
                        logger.info(f"Found buffer: {buf_name} for parameter {n} (pid={_pid})")
                    return buf_name
            raise NotImplementedError(f"No buffer found in the same module as parameter {n} (pid={_pid})")
    raise NotImplementedError(f"Parameter with id {_pid} not found")


def convert_Theta_bl_t_dict_keys(GVM: GlobalVarsManager, model: VisionTransformer, Theta_bl_t_dict: OrderedDict[int, Tensor], verbose: bool = True) -> OrderedDict[str, Tensor]:
    if verbose:
        logger.info(f"Convert Theta_bl_t_dict keys")

    new_Theta_bl_t_dict: OrderedDict[str, Tensor] = OrderedDict()
    for _pid, Theta_bl_t in Theta_bl_t_dict.items():
        buffer_name = _find_buffer_with_pid(_pid, 'Theta_bl_t', model)
        buffer_tensor: Tensor = model.get_buffer(buffer_name)
        assert buffer_tensor.shape == Theta_bl_t.shape, f"{buffer_tensor.shape} != {Theta_bl_t.shape}"
        new_Theta_bl_t_dict[buffer_name] = Theta_bl_t
    assert len(new_Theta_bl_t_dict) == len(Theta_bl_t_dict), f"{len(new_Theta_bl_t_dict)} != {len(Theta_bl_t_dict)}"
    if verbose:
        logger.info(f"Convert Theta_bl_t_dict keys done, {len(new_Theta_bl_t_dict)} keys")

    return new_Theta_bl_t_dict


def distribute_solution_Theta_bl_t(GVM: GlobalVarsManager, model: VisionTransformer, Theta_bl_t_dict: OrderedDict[str, Tensor], verbose: bool = True):
    if verbose:
        logger.info(f"Distribute solution Theta_bl_t")
    _func_start = ttime()

    for _key, Theta_bl_t in Theta_bl_t_dict.items():
        buffer_name = _key
        buffer_tensor: Tensor = model.get_buffer(buffer_name)
        assert buffer_tensor.shape == Theta_bl_t.shape, f"{buffer_tensor.shape} != {Theta_bl_t.shape}"
        buffer_tensor.copy_(Theta_bl_t.to(device=buffer_tensor.device, dtype=buffer_tensor.dtype))  # 将Theta_bl_t分配到对应的Buffer中

    if verbose:
        logger.info(f"Distribute solution {Theta_bl_t_dict.keys()} done, time: {ttime() - _func_start:.2f}s")
    return model


def set_model_flag(model: VisionTransformer, flag: Literal['use_Theta_bl_t_for_infer', 'use_Theta_h_t_for_infer'], value: bool, verbose: bool = True):
    for n, m in model.named_modules():
        if hasattr(m, flag):
            setattr(m, flag, value)
            if verbose:
                logger.info(f"Set {flag} to {value} for {n}")


def get_head_interm_tensors(GVM: GlobalVarsManager, model: VisionTransformer, idevice: str = 'cpu', idtype=torch.float32, verbose: bool = True) -> dict[str, Tensor]:
    args = GVM.args
    logger.info(f"Running the forward process.")
    _func_start = ttime()

    model = set_model_mode(GVM, model, training=False)
    torch.cuda.empty_cache()

    min_samples = model.embed_dim if args.theta_h_min_samples == 0 else args.theta_h_min_samples
    max_samples = args.eval_batch_size * args.theta_h_batches
    assert max_samples >= min_samples, \
        f"theta_h capacity {max_samples} is smaller than theta_h_min_samples={min_samples}"
    dataset, sample_info = define_statistics_dataset(
        GVM, GVM.cl_mngr.current_task_classes, transform_type=args.transform_type,
        min_samples=min_samples, max_samples=max_samples,
        target_map_to_local=args.seperate_head,
        seed=args.seed + GVM.cl_mngr.current_taskid * 1009 + 23)
    assert len(dataset) >= min_samples
    cumulative_samples = GVM.cache_dict.get('theta_h_effective_samples', 0) + len(dataset)
    logger.info(f"[CUO Sampling][Head][Task {GVM.cl_mngr.current_taskid + 1}] {sample_info}, "
                f"cumulative_samples={cumulative_samples}, feature_dim={model.embed_dim}, "
                f"N_over_D={cumulative_samples / model.embed_dim:.3f}")
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    embed_dim = model.embed_dim
    DEVICE = torch.device(idevice)
    feats = torch.empty([len(dataset), embed_dim], device=DEVICE, dtype=idtype)
    labels = torch.empty([len(dataset)], device=DEVICE, dtype=torch.long)

    head_w = model.get_parameter('head.weight').detach().T.to(device=DEVICE, dtype=idtype)  # [D, C]

    smp_idx = 0
    for img, lbl in dataloader:
        img: Tensor
        with torch.no_grad():
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True)  # [B, D]
        _feat = _feat.to(device=DEVICE, dtype=idtype)
        _b = _feat.shape[0]
        feats[smp_idx:smp_idx + _b] = _feat
        labels[smp_idx:smp_idx + _b] = lbl.to(device=DEVICE)
        smp_idx += _b
    assert smp_idx == len(dataset), f"{smp_idx} != {len(dataset)}"
    assert feats.shape == (len(dataset), model.embed_dim)
    GVM.cache_dict['theta_h_effective_samples'] = cumulative_samples

    head_tensor_dict: dict[str, Tensor] = OrderedDict()
    head_tensor_dict['head_w'] = head_w  # [D, C]
    head_tensor_dict['feats'] = feats  # [N, D]
    head_tensor_dict['labels'] = labels  # [N]

    logger.info(f"Running the forward process done ({len(dataset)} samples), time: {ttime() - _func_start:.2f}s")
    return head_tensor_dict


def calc_and_store_head_solution_components(GVM: GlobalVarsManager, head_tensor_dict: dict[str, Tensor], idevice: str = 'cpu', idtype=torch.float32):
    logger.info(f"Calc head solution components")
    args = GVM.args
    _func_start = ttime()
    DEVICE = torch.device(idevice)

    Wi: Tensor = head_tensor_dict['head_w'].to(device=DEVICE, dtype=idtype)  # [D, C]
    Xi: Tensor = head_tensor_dict['feats'].to(device=DEVICE, dtype=idtype)  # [N, D]
    Li: Tensor = head_tensor_dict['labels'].to(device=DEVICE, dtype=torch.long)  # [N]

    _all_head_w_list: list[Tensor] = []
    _all_head_b_list: list[Tensor] = []
    logger.info(f"GVM.param_dict.keys(): {GVM.param_dict.keys()}")
    for _t in range(GVM.cl_mngr.current_taskid + 1):
        _all_head_w_list.append(GVM.param_dict[f'task_params_{_t}']['head.weight'].detach().T.to(device=DEVICE, dtype=idtype))
        _all_head_b_list.append(GVM.param_dict[f'task_params_{_t}']['head.bias'].detach().T.to(device=DEVICE, dtype=idtype))
    assert torch.equal(_all_head_w_list[-1], Wi), "Head weight mismatch"
    Zij = [Xi @ _wj + _bj for _wj, _bj in zip(_all_head_w_list, _all_head_b_list)]

    for _zij in Zij[:-1]:
        _zij.mul_(args.theta_h_alpha)
    Zij[-1].copy_(F.one_hot(Li, num_classes=len(GVM.cl_mngr.current_task_classes)).to(dtype=idtype) * args.theta_h_beta)

    Pij = [Xi.T @ _zij for _zij in Zij]
    cov_Xi = Xi.T @ Xi  # [D, D]

    if GVM.cl_mngr.current_taskid > 0:
        prev_Ft = GVM.cache_dict['Theta_h_t-cums']['Ft'].to(device=DEVICE, dtype=idtype)
        _sum_Zij_i_lt_j = prev_Ft @ Wi * args.theta_h_alpha

    if 'Theta_h_t-cums' not in GVM.cache_dict:
        GVM.cache_dict['Theta_h_t-cums'] = {}
        GVM.cache_dict['Theta_h_t-cums']['Ft'] = cov_Xi.cpu().float()
        GVM.cache_dict['Theta_h_t-cums']['Wt'] = [Wi.cpu().float()]
        GVM.cache_dict['Theta_h_t-cums']['Pt'] = [_pij.cpu().float() for _pij in Pij]  # [D, C]
    else:
        GVM.cache_dict['Theta_h_t-cums']['Ft'] += cov_Xi.cpu().float()  # [D, D]
        GVM.cache_dict['Theta_h_t-cums']['Wt'].append(Wi.cpu().float())  # [D, C]
        assert len(GVM.cache_dict['Theta_h_t-cums']['Pt']) == len(Pij) - 1
        for _Pt, _pij in zip(GVM.cache_dict['Theta_h_t-cums']['Pt'], Pij[:-1]):
            _Pt += _pij.cpu().float()  # [D, C]
        GVM.cache_dict['Theta_h_t-cums']['Pt'].append((_sum_Zij_i_lt_j + Pij[-1]).cpu().float())  # [D, C]
        assert len(GVM.cache_dict['Theta_h_t-cums']['Pt']) == GVM.cl_mngr.current_taskid + 1, f"{len(GVM.cache_dict['Theta_h_t-cums']['Pt'])} != {GVM.cl_mngr.current_taskid + 1}"

    head_tensor_dict.pop('labels', None)
    head_tensor_dict.pop('feats', None)

    logger.info(f"Calc head solution components done, time: {ttime() - _func_start:.2f}s")


def calc_solution_Theta_h_t(GVM: GlobalVarsManager, eps: float = 1e-5, idevice: str = 'cpu', idtype=torch.float32) -> Tensor:
    logger.info(f"Calc solution H")
    _func_start = ttime()
    DEVICE = torch.device(idevice)

    if 'Theta_h_t-cums' not in GVM.cache_dict:
        raise NotImplementedError("!!")
    head_solu_comp_dict: dict[str, Tensor] = GVM.cache_dict['Theta_h_t-cums']

    Ft: Tensor = head_solu_comp_dict['Ft'].to(device=DEVICE, dtype=idtype)  # [D, D]
    Wt: list[Tensor] = head_solu_comp_dict['Wt']  # [D, C]*T
    Qt: Tensor = torch.cat([Ft @ _Wti.to(device=DEVICE, dtype=idtype) for _Wti in Wt], dim=1)  # [D, C*T]
    Pt: Tensor = torch.cat(head_solu_comp_dict['Pt'], dim=1).to(device=DEVICE, dtype=idtype)  # [D, C*T]

    assert Qt.shape[1] == Pt.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{Qt.shape[1]} != {Pt.shape[1]} != {len(GVM.cl_mngr.sofar_task_classes)}"
    assert Ft.device == Qt.device == Pt.device, f"{Ft.device} != {Qt.device} != {Pt.device}"

    Ft = Ft.to(torch.float64)
    Ft.diagonal().add_(eps)
    Theta_h_t = torch.linalg.solve(Ft, Pt.to(torch.float64) - Qt.to(torch.float64))
    Theta_h_t = Theta_h_t.to(dtype=torch.float32, device='cpu')  # [D, C*T]
    GVM.cache_dict['Theta_h_t'] = Theta_h_t
    logger.info(f"Calc solution Theta_h_t done, time: {ttime() - _func_start:.2f}s")
    return Theta_h_t


def distribute_solution_Theta_h_t(GVM: GlobalVarsManager, model: VisionTransformer, Theta_h_t: Tensor):
    logger.info(f"Distribute solution Theta_h_t")
    _func_start = ttime()

    model.register_buffer('Theta_h_t', Theta_h_t.T, persistent=False)  # [D, C*T] -> [C*T, D]

    logger.info(f"Distribute solution Theta_h_t done, time: {ttime() - _func_start:.2f}s")
    return model
