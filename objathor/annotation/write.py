from importlib import import_module
from typing import Union, Callable, Any, Dict, Optional, cast

import compress_json
import compress_pickle


def write(
    anno: Dict[str, Any],
    output_file: Union[str, Callable[[Dict[str, Any]], Optional[Any]]],
    **kwargs: Any,
) -> None:
    if isinstance(output_file, str):
        if output_file.endswith(".json.gz"):
            compress_json.dump(anno, output_file, json_kwargs=dict(indent=2))
        elif output_file.endswith(".pickle.gz") or output_file.endswith(".pkl.gz"):
            compress_pickle.dump(anno, output_file)
        else:
            try:
                module_name, function_name = output_file.rsplit(".", 1)
                getattr(import_module(module_name), function_name)(anno, **kwargs)
            except Exception as e:
                print("Error", e)
                raise NotImplementedError(
                    "Only .pkl.gz and .json.gz supported, besides appropriate library function identifiers"
                )
    elif isinstance(output_file, Callable):
        output_file(anno)
    else:
        raise NotImplementedError(f"Unsupported output_file arg of type {type(output_file).__name__}")
