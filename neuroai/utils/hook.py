from .getting_modules import get_layer_by_name

class ForwardHook:
    def __init__(self, model, hook_layer_name: str):
        layer = get_layer_by_name(model=model, layer_name=hook_layer_name)
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