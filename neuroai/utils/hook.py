from .getting_modules import get_layer_by_name
import torch

class ForwardHook:
    def __init__(self, model, hook_layer_name: str):
        layer = get_layer_by_name(model=model, layer_name=hook_layer_name)
        assert isinstance(layer, torch.nn.Conv2d), f"Oops, you chose a layer that's not a Conv2d layer (it's: {type(layer)}). Try picking a Conv2d layer instead (feel free to call mayukh for help if you're stuck!)"
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None
        self.hook_layer_name = hook_layer_name

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

    def __repr__(self):
        return f"ForwardHook(hook_layer_name={self.hook_layer_name})"