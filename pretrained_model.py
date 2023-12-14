import torch, torchvision
import mmseg
import mmcv
import mmengine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path as osp
import numpy as np
from PIL import Image
from mmengine.runner import Runner
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


def train_model(
    data_root: str,
    img_dir: str,
    fluid_dir: str,
    config_py_path: str,
    pth_path: str,
    verbose: bool = False,
    dataset_num: int = 1,
) -> None:
    root = data_root
    img_dir = img_dir
    ann_dir = fluid_dir

    if verbose:
        print(
            f"""PyTorch Version:{torch.__version__}, 
GPU In-Use: {torch.cuda.is_available()}"""
        )
        assert torch.cuda.is_available() == True, "GPU not available"
        print(f"MMSegmentation Version:{mmseg.__version__}")

    cfg = mmengine.Config.fromfile(f"{config_py_path}")

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type="BN", requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 2
    cfg.model.auxiliary_head.num_classes = 2

    # cfg.model.decode_head.out_channels=1
    # cfg.model.auxiliary_head.out_channels=1

    # cfg.model.decode_head.loss_decode=dict(
    #     type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
    # )

    # cfg.model.auxiliary_head.loss_decode=dict(
    #     type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.4
    # )

    # Modify dataset type and path
    cfg.dataset_type = "BOE_Chiu_Dataset"
    cfg.data_root = root

    cfg.train_dataloader.batch_size = 2

    cfg.train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", scale=(512, 512), keep_ratio=True),
        dict(type="PackSegInputs"),
    ]

    cfg.test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(512, 512), keep_ratio=True),
        dict(type="LoadAnnotations"),
        dict(type="PackSegInputs"),
    ]

    if dataset_num > 1:
        for item in cfg.train_dataloader.dataset["datasets"]:
            item["type"] = cfg.dataset_type
            item["data_root"] = cfg.data_root
            item["data_prefix"] = dict(img_path=img_dir, seg_map_path=ann_dir)
            item["pipeline"] = cfg.train_pipeline
            item["ann_file"] = "splits/train.txt"
    elif dataset_num == 1:
        cfg.train_dataloader.dataset.type = cfg.dataset_type
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.train_dataloader.dataset.data_prefix = dict(
            img_path=img_dir, seg_map_path=ann_dir
        )
        cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
        cfg.train_dataloader.dataset.ann_file = "splits/train.txt"
    else:
        raise ValueError(
            "Check the Number of Datasets under train_dataloader.dataset in the Config File. "
        )

    cfg.val_dataloader.dataset.type = cfg.dataset_type
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix = dict(
        img_path=img_dir, seg_map_path=ann_dir
    )
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    cfg.val_dataloader.dataset.ann_file = "splits/val.txt"

    cfg.test_dataloader = cfg.val_dataloader

    # Define Evaluator
    cfg.test_evaluator = ["mDice", "mIoU", "mFscore"]
    cfg.val_evaluator.iou_metrics = ["mDice", "mIoU", "mFscore"]

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = f"{pth_path}"

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work_dirs/tutorial"

    cfg.train_cfg.max_iters = 200
    cfg.train_cfg.val_interval = 200
    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint.interval = 200

    # Set seed to facitate reproducing the result
    cfg.randomness = dict(seed=0)

    # Let's have a look at the final config used for training
    if verbose:
        print(f"Config:\n{cfg.pretty_text}")

    runner = Runner.from_cfg(cfg)

    runner.train()
