import sys, os

from .PinkModule.logging import PinkBlackLogger
from omegaconf import OmegaConf, DictConfig


def from_cli(default_config: DictConfig):
    default_config.merge_with_cli()
    if "gpu" in default_config.keys():
        # Default argument로 gpu를 줬다면 이렇게 세팅
        os.environ.update({"CUDA_VISIBLE_DEVICES": str(default_config.gpu)})
    return default_config


def setup(
    trace=False,
    pdb_on_error=True,
    default_config=None,
    autolog=False,
    autolog_dir="pinkblack_autolog",
):
    """
    :param trace:
    :param pdb_on_error:
    :param default_config: dict or str(yaml file)
    :param autolog:
    :param autolog_dir:
    :return: config
    """
    if trace:
        import backtrace

        backtrace.hook(align=True)

    if pdb_on_error:
        old_hook = sys.excepthook

        def new_hook(type_, value, tb):
            old_hook(type_, value, tb)
            if type_ != KeyboardInterrupt:
                import pdb

                pdb.post_mortem(tb)

        sys.excepthook = new_hook

    args = None
    if default_config is not None:
        if isinstance(default_config, str):
            default_config = OmegaConf.load(default_config)
        elif isinstance(default_config, dict):
            default_config = OmegaConf.create(default_config)
        args = from_cli(default_config)

    import time, datetime

    dt = datetime.datetime.fromtimestamp(time.time())
    dt = datetime.datetime.strftime(dt, f"{os.path.basename(sys.argv[0])}.%Y%m%d_%H%M%S.log")

    if args is not None and hasattr(args, "ckpt"):
        logpath = args.ckpt + "_" + dt
    else:
        logpath = os.path.join(autolog_dir, dt)

    os.makedirs(os.path.dirname(logpath), exist_ok=True)

    if args is not None:
        conf = OmegaConf.create(args.__dict__)
        conf.save(logpath[:-4] + ".yaml")
        conf.save(args.ckpt + ".yaml")

    if autolog:
        fp = open(logpath, "w")
        sys.stdout = PinkBlackLogger(fp, sys.stdout)
        sys.stderr = PinkBlackLogger(fp, sys.stderr)

    return args


def set_seeds(seed, strict=False):
    """
    strict 가 True이면, Cudnn까지도 deterministic 하게 한다.
    cudnn은 아주 조금 stochastic한 연산 결과를 보여주므로, 정확한 재현이 필요하다면 True로 설정.

    If strict == True, then cudnn backend will be deterministic
    torch.backends.cudnn.deterministic = True
    """
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
