import sys, os
import torch

def get_args(default_args: dict):
    import argparse
    parser = argparse.ArgumentParser()
    for k, v in default_args.items():
        k = k.lower()
        parser.add_argument(f'--{k}', default=v, help=f'{k} : default:{v}')
    args = parser.parse_args()

    if hasattr(args, "gpu"):
        os.environ.update({'CUDA_VISIBLE_DEVICES': str(args.gpu)})

    return args


def setup(trace=True, pdb_on_error=True, default_args=None):
    """
    :param trace:
    :param pdb_on_error:
    :param default_args:
    gpu -> CUDA_VISIBLE_DEVICES
    :return: argparsed args

    Example >>
    ```python3
    # Before PinkBlack
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ckpt', default='ckpt.pth', type=str)
    args = parser.parse_args()
    ```
    ```bash
    CUDA_VISIBLE_DEVICES=1,3 python myscript.py --batch_size 32 --ckpt ckpt.pth --epochs 100 --lr 0.001
    ```
    ```python3
    # Using PinkBlack
    PinkBlack.io.setup(trace=True, pdb_on_error=True, default_args={'gpu':"1,3", 'batch_size':32, 'lr':1e-3, 'epochs':100, 'ckpt': "ckpt.pth"})
    ```
    ```bash
    python myscript.py --gpu 1,3 --batch_size 32 --ckpt ckpt.pth --epochs 100 --lr 0.001
    ```

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
    if default_args is not None:
        args = get_args(default_args)

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict:
            torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(net, path, optimizer=None):
    """
    :param net: nn.Module
    :param path: "/data/checkpoint/module.pth"
    :param optimizer: Optional
    :return: net, path, optimizer(if it exists)
    """

    saved_dict = torch.load(path)

    if 'state_dict' in saved_dict:
        net.load_state_dict(saved_dict['state_dict'])
        del saved_dict['state_dict']
    elif 'net.state_dict' in saved_dict:
        net.load_state_dict(saved_dict['net.state_dict'])
        del saved_dict['net.state_dict']
    else:
        raise ValueError("No networks are saved in the checkpoint file.")

    if optimizer is not None and 'optimizer.state_dict' in saved_dict:
        optimizer.load_state_dict(saved_dict['optimizer.state_dict'])
        del saved_dict['optimizer.state_dict']

    if optimizer is None and 'optimizer.state_dict' in saved_dict:
        del saved_dict['optimizer.state_dict']

    print("Loaded", saved_dict)
    if optimizer is None:
        return net, saved_dict
    else:
        return net, saved_dict, optimizer


load_model = load_checkpoint


def save_checkpoint(obj, path):
    """
    :param obj: dict or nn.Module
    :param path: path to save
    """
    if isinstance(obj, dict):
        torch.save(obj, path)
    elif isinstance(obj, torch.nn.Module):
        if hasattr(obj, 'module'):
            save_dict = {'state_dict': obj.module.state_dict()}
        else:
            save_dict = {'state_dict': obj.state_dict()}
        torch.save(save_dict)
    else:
        raise NotImplementedError

save_model = save_checkpoint


