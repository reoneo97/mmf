#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import logging
import random
import typing

import torch
from mmf.common.registry import registry
from mmf.utils.build import build_config, build_trainer
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import distributed_init, get_rank, infer_init_method, is_xla
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.flags import flags
from mmf.utils.general import log_device_names
from mmf.utils.logger import setup_logger, setup_very_basic_config
from mmf.common.sample import to_device

setup_very_basic_config()
from torch.autograd import Variable


def main(configuration, init_distributed=False, predict=True):
    # A reload might be needed for imports
    setup_imports()
    configuration.import_user_dir()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    if init_distributed:
        distributed_init(config)

    # print('I am not Distributed')
    seed = config.training.seed
    config.training.seed = set_seed(seed if seed == -1 else seed + get_rank())
    registry.register("seed", config.training.seed)

    config = build_config(configuration)
    setup_logger(
        color=config.training.colored_logs, disable=config.training.should_not_log
    )
    logger = logging.getLogger("mmf_cli.run")
    logger.info(config.dataset_config)
    # Log args for debugging purposes
    logger.info(configuration.args)
    logger.info(f"Torch version: {torch.__version__}")
    log_device_names()
    logger.info(f"Using seed {config.training.seed}")

    trainer = build_trainer(config)
    trainer.load()
    dl = trainer.train_loader
    dataset_loader = trainer.dataset_loader
    dataset_type='train'
    reporter = dataset_loader.get_test_reporter(dataset_type)
    reporter.get_dataloader()
    model = trainer.model
    for idx, batch in enumerate(dl):
        prepared_batch = reporter.prepare_batch(batch)
        targets = prepared_batch['targets']
        prepared_batch = to_device(prepared_batch, trainer.device)
        prepared_batch['image'] = Variable(prepared_batch['image'], requires_grad=True)
        
        model_output = model(prepared_batch)
        scores = model_output['scores'][torch.arange(len(targets)), targets]
        score_scalar = scores.sum()
        score_scalar.backward()
        x_grad = prepared_batch['image'].grad
        prepared_batch.detach()
        saliency = torch.max(torch.abs(x_grad),1)[0]
        print('Saliency:', saliency.shape)
        show_saliency_maps(prepared_batch['image'], targets, saliency, idx)
        # break

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import numpy as np

def deprocess(img, should_rescale=True):
    SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    transform = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
    ])
    return transform(img)



def show_saliency_maps(X, y, saliency, idx, class_names=['Not Hateful', 'Hateful']):
    # Convert X and y from numpy arrays to Torch Tensors
    X = deprocess(X)
    print(X.min(), X.max())

    # print(type(X))
    X = X.cpu().numpy()
    print(X.shape)

    # Compute saliency maps for images in X
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.cpu().numpy()

    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i].transpose(1,2,0))
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.gray)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.savefig(f'saliency/saliency_map_{idx}.png', bbox_inches = 'tight')


def distributed_main(device_id, configuration, predict=False):
    config = configuration.get_config()
    config.device_id = device_id

    if config.distributed.rank is None:
        config.distributed.rank = config.start_rank + device_id

    main(configuration, init_distributed=True, predict=predict)


def run(opts: typing.Optional[typing.List[str]] = None, predict: bool = False):
    """Run starts a job based on the command passed from the command line.
    You can optionally run the mmf job programmatically by passing an optlist as opts.

    Args:
        opts (typing.Optional[typing.List[str]], optional): Optlist which can be used.
            to override opts programmatically. For e.g. if you pass
            opts = ["training.batch_size=64", "checkpoint.resume=True"], this will
            set the batch size to 64 and resume from the checkpoint if present.
            Defaults to None.
        predict (bool, optional): If predict is passed True, then the program runs in
            prediction mode. Defaults to False.
    """
    setup_imports()

    if opts is None:
        parser = flags.get_parser()
        args = parser.parse_args()
    else:
        args = argparse.Namespace(config_override=None)
        args.opts = opts

    configuration = Configuration(args)
    # Do set runtime args which can be changed by MMF
    configuration.args = args
    config = configuration.get_config()
    # print(config)
    config.start_rank = 0
    if config.distributed.init_method is None:
        infer_init_method(config)

    if config.distributed.init_method is not None:
        if torch.cuda.device_count() > 1 and not config.distributed.no_spawn:
            config.start_rank = config.distributed.rank
            config.distributed.rank = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration, predict),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(0, configuration, predict)
    elif config.distributed.world_size > 1:
        if is_xla():
            import torch_xla.distributed.xla_multiprocessing as xmp

            torch.multiprocessing.set_sharing_strategy("file_system")
            xmp.spawn(
                fn=distributed_main,
                args=(configuration, predict),
                nprocs=8,  # use all 8 TPU cores
                start_method="fork",
            )
        else:
            assert config.distributed.world_size <= torch.cuda.device_count()
            port = random.randint(10000, 20000)
            config.distributed.init_method = f"tcp://localhost:{port}"
            config.distributed.rank = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration, predict),
                nprocs=config.distributed.world_size,
            )
    else:
        print('I am here')
        config.device_id = 0
        main(configuration, predict=predict)


if __name__ == "__main__":
    run()
