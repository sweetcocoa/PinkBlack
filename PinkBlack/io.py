import sys, os
import torch
import random
import numpy as np


def setup(trace=True, pdb_on_error=True, default_args=None):
    """
    커맨드라인 파싱 -> os.environ 에 추가
    gpu -> CUDA_VISIBLE_DEVICES
    :param trace:
    :param pdb_on_error:
    :param default_args: dict.
    :return:
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

    if default_args is not None:
        args = dict(default_args)
        if 'gpu' in default_args:
            args['CUDA_VISIBLE_DEVICES'] = args['gpu']
            del args['gpu']
        os.environ.update(default_args)

    new = {}

    for token in sys.argv[1:]:

        idx = token.find('=')
        if idx == -1:
            continue
        else:
            key = token[:idx]
            value = token[idx + 1:]
            if key.lower() == "gpu":
                key = "CUDA_VISIBLE_DEVICES"
            new[key] = value

    os.environ.update(new)


def set_seeds(seed, strict=False):
    """
    strict 가 True이면, Cudnn까지도 deterministic 하게 한다.
    cudnn은 아주 조금 stochastic한 연산 결과를 보여주므로, 정확한 재현이 필요하다면 True로 설정.

    If strict == True, then cudnn backend will be deterministic
    torch.backends.cudnn.deterministic = True
    """
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


def save_checkpoint(save_dict, path, is_best):
    torch.save(save_dict, path)
    if is_best:
        torch.save(save_dict, path + ".best")


save_model = save_checkpoint


