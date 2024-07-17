import torch
import torch.optim as opt


MODEL_CONFIGS = {
    'det': {
        'yolox': ['nano', 'tiny', 'small', 'medium', 'large'],
        'yolov8': ['nano', 'small', 'medium', 'large'],
    },
    'seg': {
        'yolov8': ['nano', 'small', 'medium', 'large'],
    }
}



def model_options(model_type):
    if model_type.startswith('yolox'):
        from .yolox.config import get_model_options
    elif model_type.startswith('yolov8'):
        from .yolov8.config import get_model_options
    else:
        raise ImportError('Invalid model type')
    return get_model_options()


def cal_model_uid(model_type, options):
    if model_type.startswith('yolox'):
        from .yolox.config import get_model_uid
    elif model_type.startswith('yolov8'):
        from .yolov8.config import get_model_uid
    else:
        raise ImportError('Invalid model type')
    return get_model_uid(options)


def default_options(model_type):
    if model_type.startswith('yolox'):
        from .yolox.config import get_default_options
    elif model_type.startswith('yolov8'):
        from .yolov8.config import get_default_options
    else:
        raise ImportError('Invalid model type')

    return get_default_options()[model_type]


def recomend_options(model_type):
    if model_type.startswith('yolox'):
        from .yolox.config import get_recoment_options
    elif model_type.startswith('yolov8'):
        from .yolov8.config import get_recoment_options
    else:
        raise ImportError('Invalid model type')
    return get_recoment_options()


def post_process(model_type):
    if model_type.startswith('yolox'):
        from .yolox.predictor import postprocess
    elif model_type.startswith('yolov8'):
        from .yolov8.predictor import postprocess
    else:
        raise ImportError('Invalid model type')
    return postprocess


def create_predictor(config, model_type, train_dir,
                     score_thresh=0.5,
                     iou_thresh=0.5):
    if model_type.startswith('yolox'):
        from .yolox.predictor import Predictor
    elif model_type.startswith('yolov8'):
        from .yolov8.predictor import Predictor
    else:
        raise ImportError('Invalid model type')
    return Predictor(config, model_type, train_dir,
                     score_thresh=score_thresh,
                     iou_thresh=iou_thresh)


def create_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.
        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(opt, name, opt.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = opt.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = opt.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        return optimizer



