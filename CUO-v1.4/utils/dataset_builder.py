import os
import os.path as osp
import random
from os.path import join, exists, realpath, dirname
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from typing import Union, Callable
from copy import deepcopy
from functools import partial
from loguru import logger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ImageFolder 风格的数据集,按类筛选  

class ImagePathDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.class_imgpath_dict: dict[int, str] = {}
        self.class_int_str_map: dict[int, str] = {}
        self.split_path = ""

    def __getitem__(self, index):
        return self.class_imgpath_dict[index]

    def __len__(self):
        return len(self.class_imgpath_dict)

    @property
    def class_list(self):
        return sorted(list(self.class_imgpath_dict.keys()))

    @property
    def num_classes(self):
        return len(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: image_dir=\"{self.split_path}\", full_dataset_classes={len(self)}"


class CIFAR100Path(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/cifar100_classnames.txt")):
            with open(_rf, 'r') as f:
                for _i, line in enumerate(f.readlines()):
                    _dn = line.strip('\n')
                    name_map[_i] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 100

        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_int]

        assert len(self.class_imgpath_dict) == 100


class ImageNetRPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/imagenet_r_classnames.txt")):
            with open(_rf, 'r') as f:
                for line in f.readlines():
                    _sn = line.split(' ')[0]
                    _dn = line.strip('\n')[len(_sn) + 1:]
                    name_map[_sn] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 200

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_dir]

        assert len(self.class_imgpath_dict) == 200


class SDomainNetPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_int

        assert len(self.class_imgpath_dict) == 200


class ImageNetAPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_dir

        assert len(self.class_imgpath_dict) == 200


class ImagePathDatasetClassManager():
    def __init__(self, **kwargs):
        datadir_root = realpath(osp.expanduser("~/data"))
        self.dataset_dict = {
            'cifar100': partial(CIFAR100Path, root_dir=join(datadir_root, 'data.CIFAR100') if not (v := kwargs.get('cifar100')) else v),
            'imagenet_r': partial(ImageNetRPath, root_dir=join(datadir_root, 'data.ImageNet-R-coda') if not (v := kwargs.get('imagenet_r')) else v),
            'sdomainet': partial(SDomainNetPath, root_dir=join(datadir_root, 'data.DomainNet') if not (v := kwargs.get('sdomainet')) else v),
            'imagenet_a': partial(ImageNetAPath, root_dir=join(datadir_root, 'data.ImageNet-A') if not (v := kwargs.get('imagenet_a')) else v),
        }

    def __getitem__(self, dataset: str):
        dataset = dataset.lower()
        if dataset not in (_valid_names := self.dataset_dict.keys()):
            raise NameError(f"{dataset} is not in {_valid_names}")
        return self.dataset_dict[dataset]


class ClassIncremantalDataset(Dataset):
    def __init__(self, path_dataset: ImagePathDataset, task_class_list: list[int], transforms: T.Compose = None, target_transforms: Callable = None, expand_times: int = 1, verbose: bool = False, return_index: bool = False, sample_type='path'):
        super().__init__()
        self.path_dataset = path_dataset
        self.task_class_list = tuple(deepcopy(task_class_list))
        assert isinstance(expand_times, int) and expand_times >= 1
        self.expand_times = expand_times
        self.transforms = transforms
        self.target_transforms = target_transforms
        assert sample_type in ('path', 'image')
        self.sample_type = sample_type

        # FIXME:
        self.samples, self.labels = self.get_all_samples(sample_type=self.sample_type)
        self.verbose = verbose
        self.return_index = return_index
        self.num_samples = len(self.labels)

        self.cache_dict = {}

    def get_all_samples(self, sample_type: str = 'path') -> tuple[list[Image.Image | str], list[int]]:
        assert sample_type in ('path', 'image'), f"{sample_type}"
        smp_list = []
        lbl_list = []
        for cls_int in self.task_class_list:
            assert cls_int in self.path_dataset.class_list
            assert len(self.path_dataset[cls_int]) > 0
            for img_path in sorted(self.path_dataset[cls_int]):
                if sample_type == 'image':
                    sample: Image.Image = Image.open(img_path).convert('RGB')
                elif sample_type == 'path':
                    sample: str = img_path
                smp_list.append(sample)

                label = cls_int
                if self.target_transforms is not None:
                    label = self.target_transforms(label)
                lbl_list.append(label)

        return smp_list, lbl_list

    def read_one_image_label(self, index: int) -> tuple[Image.Image, int]:
        if self.sample_type == 'path':
            if self.expand_times == 1:
                img = Image.open(self.samples[index]).convert('RGB')
            elif self.expand_times > 1:
                img = Image.open(self.samples[index]).convert('RGB')
                # if index not in self.cache_dict:
                #     img = Image.open(self.samples[index]).convert('RGB')
                #     self.cache_dict[index] = img
                # else:
                #     # img = deepcopy(self.cache_dict[index])
                #     img = self.cache_dict[index]
            else:
                raise ValueError(f"{self.expand_times}")

            lbl = self.labels[index]
            return img, lbl
        elif self.sample_type == 'image':
            return self.samples[index], self.labels[index]
        else:
            raise NameError(f"{self.sample_type}")

    def __getitem__(self, index: int) -> tuple[Union[Image.Image, Tensor], int]:
        index %= self.num_samples
        img, lbl = self.read_one_image_label(index)

        if self.transforms is not None:
            if self.return_index:
                return self.transforms(img), lbl, index
            else:
                return self.transforms(img), lbl
        if self.return_index:
            return img, lbl, index
        else:
            return img, lbl

    def __len__(self):
        return self.num_samples * self.expand_times

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__} for {self.path_dataset.__repr__()}: task_class_list({len(self.task_class_list)})={self.task_class_list}, num_samples={len(self.samples)}, expand_times={self.expand_times}"
        if self.verbose:
            _repr += f", \nclasses_string={[self.path_dataset.class_int_str_map[cls_int] for cls_int in self.task_class_list]}"
            _repr += f", \ntransform={self.transforms}"
            _repr += f", \ntarget_transforms={self.target_transforms}"
            _repr += f", \nsample_type={self.sample_type}"
            _repr += f"."
        return _repr


class TargetTransform():
    def __init__(self, label_map_g2l: dict[int, tuple[int, int, int]], target_map_to_local: bool) -> None:
        self.label_map_g2l = deepcopy(label_map_g2l)  # {original_label: (taskid, local_label, global_label)}
        self.target_map_to_local = target_map_to_local

    def __call__(self, target: int):
        if self.target_map_to_local:
            return self.label_map_g2l[target][1]
        else:
            return self.label_map_g2l[target][2]

    def __repr__(self) -> str:
        label_map = {k: v[1] if self.target_map_to_local else v[2] for k, v in self.label_map_g2l.items()}
        _repr = str(label_map)
        return _repr


class StatisticsDataset(Dataset):
    """One deterministic view per selected image plus stochastic augmented views."""

    def __init__(self, source_dataset: ClassIncremantalDataset, base_indices: list[int], augment_indices: list[int], augment_transform):
        super().__init__()
        self.source_dataset = source_dataset
        self.base_indices = list(base_indices)
        self.augment_indices = list(augment_indices)
        self.base_transform = source_dataset.transforms
        self.augment_transform = augment_transform
        assert len(self.source_dataset) == self.source_dataset.num_samples
        assert len(self.base_indices) + len(self.augment_indices) > 0
        assert all(0 <= idx < self.source_dataset.num_samples for idx in self.base_indices + self.augment_indices)
        assert not self.augment_indices or self.augment_transform is not None

    def __getitem__(self, index: int):
        if index < len(self.base_indices):
            source_index = self.base_indices[index]
            transform = self.base_transform
        else:
            source_index = self.augment_indices[index - len(self.base_indices)]
            transform = self.augment_transform
        img, label = self.source_dataset.read_one_image_label(source_index)
        return transform(img), label

    def __len__(self):
        return len(self.base_indices) + len(self.augment_indices)


def _build_transform(GVM, training: bool, transform_type: str, use_eval_transform: bool = False):
    """Build the same train/eval transform used by ``define_dataset``."""
    if GVM.args.interp_mode == 'auto':
        match GVM.args.dataset:
            case 'imagenet_r' | 'sdomainet' | 'imagenet_a':
                GVM.args.interp_mode = 'bilinear'
            case 'cifar100':
                GVM.args.interp_mode = 'bicubic'
            case _:
                raise NotImplementedError(GVM.args.dataset)

    match GVM.args.interp_mode:
        case 'bilinear':
            interp_mode = T.InterpolationMode.BILINEAR
        case 'bicubic':
            interp_mode = T.InterpolationMode.BICUBIC
        case _:
            raise ValueError(GVM.args.interp_mode)

    is_training = training and not use_eval_transform
    match transform_type:
        case 'timm':
            return create_transform(**resolve_data_config(GVM.cache_dict['pretrained_cfg']), is_training=is_training)
        case 'autoaug':
            dmean: tuple[float] = GVM.cache_dict['pretrained_cfg']['mean']
            dstd: tuple[float] = GVM.cache_dict['pretrained_cfg']['std']
            bilinear = T.InterpolationMode.BILINEAR
            if is_training:
                match GVM.args.dataset:
                    case 'cifar100':
                        return T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet' | 'imagenet_a':
                        return T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(GVM.args.dataset)
            match GVM.args.dataset:
                case 'cifar100':
                    return T.Compose([T.Resize((224, 224), antialias=True, interpolation=interp_mode), T.ToTensor(), T.Normalize(dmean, dstd)])
                case 'imagenet_r' | 'sdomainet' | 'imagenet_a':
                    return T.Compose([T.Resize((256, 256), antialias=True, interpolation=interp_mode), T.CenterCrop(224), T.ToTensor(), T.Normalize(dmean, dstd)])
                case _:
                    raise NotImplementedError(GVM.args.dataset)
        case _:
            raise NotImplementedError(transform_type)


def define_dataset(GVM, task_classes: list[int], training: bool, transform_type: str = 'timm', target_map_to_local: bool = True,
                   use_eval_transform: bool = False, expand_times: int = 1, verbose: bool = False, **kwargs) -> ClassIncremantalDataset:
    '''
    transform_type: 'timm', 'autoaug'
    '''
    transforms = _build_transform(GVM, training, transform_type, use_eval_transform)

    target_transforms = TargetTransform(GVM.label_map_g2l, target_map_to_local)

    _mode = 'train' if training else 'eval'
    dataset = ClassIncremantalDataset(GVM.path_data_dict[_mode], task_classes, transforms, target_transforms, expand_times=expand_times, verbose=verbose, return_index=False, sample_type=GVM.args.sample_type)

    if verbose:
        logger.info(f":: Current {_mode}_dataset: {str(dataset)}")

    return dataset


def define_statistics_dataset(GVM, task_classes: list[int], transform_type: str, min_samples: int, max_samples: int | None,
                              target_map_to_local: bool, seed: int, verbose: bool = False):
    """Build a fixed-size CUO statistics dataset, augmenting only a shortfall."""
    assert isinstance(min_samples, int) and min_samples >= 1
    assert max_samples is None or isinstance(max_samples, int) and max_samples >= min_samples

    source_dataset = define_dataset(GVM, task_classes, training=True, transform_type=transform_type,
                                    target_map_to_local=target_map_to_local, use_eval_transform=True,
                                    expand_times=1, verbose=False)
    original_samples = source_dataset.num_samples
    assert original_samples > 0
    target_samples = max(original_samples, min_samples)
    if max_samples is not None:
        target_samples = min(target_samples, max_samples)

    if original_samples > target_samples:
        step = original_samples / target_samples
        base_indices = [int(i * step) for i in range(target_samples)]
    else:
        base_indices = list(range(original_samples))

    num_augmented = target_samples - len(base_indices)
    augment_indices = []
    rng = random.Random(seed)
    while len(augment_indices) < num_augmented:
        shuffled = list(range(original_samples))
        rng.shuffle(shuffled)
        augment_indices.extend(shuffled[:num_augmented - len(augment_indices)])

    augment_transform = _build_transform(GVM, training=True, transform_type=transform_type) if augment_indices else None
    dataset = StatisticsDataset(source_dataset, base_indices, augment_indices, augment_transform)
    assert len(dataset) == target_samples
    info = {
        'original_samples': original_samples,
        'real_samples': len(base_indices),
        'augmented_samples': len(augment_indices),
        'effective_samples': len(dataset),
    }
    if verbose:
        logger.info(f"Statistics dataset: classes={task_classes}, {info}")
    return dataset, info
