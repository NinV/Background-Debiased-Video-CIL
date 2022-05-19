import functools


class OutputHookWrapper:
    def __init__(self, as_tensor=False):
        self.as_tensor = as_tensor
        self.output = None

    def __call__(self, module, input, output) -> None:
        if self.as_tensor:
            self.output = output
        else:
            self.output = output.detach().cpu().numpy()


class OutputHook:
    """
    based on mmaction Output hook but this class is pickle-able (required in ddp)
    """
    """Output feature map of some layers.

    Args:
        module (nn.Module): The whole module to get layers.
        outputs (tuple[str] | list[str]): Layer name to output. Default: None.
        as_tensor (bool): Determine to return a tensor or a numpy array.
            Default: False.
    """

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self._layer_outputs = {}
        self.handles = []
        self.register(module)

    def register(self, module):
        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    hook = OutputHookWrapper()
                    h = layer.register_forward_hook(hook)
                    self._layer_outputs[name] = hook
                except AttributeError:
                    raise AttributeError(f'Module {name} not found')
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def get_layer_output(self, layer_name):
        return self._layer_outputs[layer_name].output


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
