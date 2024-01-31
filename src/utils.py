import inspect
from functools import wraps


def type_checking(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        parameters = signature.parameters

        for i, arg in enumerate(args):
            param_name = list(parameters.keys())[i]
            param_type = parameters[param_name].annotation
            if param_name == 'self' or param_type is inspect._empty:
                continue
            if not isinstance(arg, param_type):
                raise TypeError(f"{func.__qualname__}: Argument "
                                f"'{param_name}' must be of type "
                                f"'{param_type}'")

        for param_name, arg in kwargs.items():
            param_type = parameters[param_name].annotation
            if param_type is inspect._empty:
                continue
            if not isinstance(arg, param_type):
                raise TypeError(f"{func.__qualname__}: Argument "
                                f"'{param_name}' must be of type "
                                f"'{param_type}'")
        return func(*args, **kwargs)

    return wrapper
