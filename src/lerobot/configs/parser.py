# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import importlib
import inspect
import json
import pkgutil
import sys
import tempfile
from argparse import ArgumentError
from collections.abc import Callable, Iterable, Sequence
from dataclasses import MISSING, Field, fields as dataclass_fields, is_dataclass
from functools import wraps
from pathlib import Path
from pkgutil import ModuleInfo
from types import ModuleType, UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import draccus
import yaml  # type: ignore[import-untyped]

from lerobot.utils.utils import has_method

F = TypeVar("F", bound=Callable[..., object])

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"

# Storage for path args extracted from YAML/JSON config files, so that
# get_path_arg() can find them even when they weren't passed via CLI.
_config_path_args: dict[str, str] = {}

# Storage for non-path YAML overrides so validate() can pass them to from_pretrained.
_config_yaml_overrides: dict[str, list[str]] = {}

# Config-file fields the user actually chose, i.e. those whose value differs from the config
# class's declared defaults, keyed by top-level field. A config file dumped by a previous run
# (draccus serializes every default) must not make each field look deliberately set, or
# checkpoint-owned adoption paths hard-error on fields nobody chose.
_config_explicit_fields: dict[str, set[str]] = {}


def _clear_config_overrides() -> None:
    """Clear all module-global state associated with a config parse."""
    _config_path_args.clear()
    _config_yaml_overrides.clear()
    _config_explicit_fields.clear()


def _flatten_to_cli_args(d: dict, prefix: str = "") -> list[str]:
    """Recursively flatten a nested dict to CLI-style args (e.g. {"lr": 1e-4} -> ["--lr=0.0001"])."""
    args = []
    for key, value in d.items():
        if key in (PATH_KEY, draccus.CHOICE_TYPE_KEY):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            value = str(value).lower()
        if isinstance(value, dict):
            args.extend(_flatten_to_cli_args(value, full_key))
        elif value is not None and not isinstance(value, list):
            args.append(f"--{full_key}={value}")
    return args


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Extract nested CLI overrides in equals or separate-value form, excluding choice/path options.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    for index, arg in enumerate(args):
        if not arg.startswith(detect_string):
            continue
        nested_arg = arg.removeprefix(detect_string)
        if nested_arg.split("=", 1)[0] in (draccus.CHOICE_TYPE_KEY, PATH_KEY):
            continue
        attr_level_args.append(f"--{nested_arg}")
        if "=" not in arg:
            # Preserve draccus's separate value tokens, including negative numbers
            # and structured values, stopping before the next unrelated option.
            for value in args[index + 1 :]:
                if value.startswith("--"):
                    break
                attr_level_args.append(value)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict[str, str]:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ["--env.discover_packages_path=my_package", "--other_arg=value"]
        >>> parse_plugin_args("discover_packages_path", args)
        {'env.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, i.e. when
    the package is imported the gym environment should be registered and the config classes
    registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")  # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg: ModuleType) -> Iterable[ModuleInfo]:
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    result = parse_arg(f"{field_name}.{PATH_KEY}", args)
    if result is None:
        result = _config_path_args.get(field_name)
    return result


def get_yaml_overrides(field_name: str) -> list[str]:
    return _config_yaml_overrides.get(field_name, [])


def get_override_field_names(overrides: Sequence[str]) -> set[str]:
    """Return explicitly overridden fields, including parents of nested fields.

    Accepts both ``--field=value`` and draccus's space-separated ``--field value``; treating
    the latter as unset would let a deliberate override be silently replaced by a
    checkpoint-owned default.
    """
    names = {
        argument.removeprefix("--").split("=", 1)[0] for argument in overrides if argument.startswith("--")
    }
    return names | {name.split(".", 1)[0] for name in names}


def _field_default(field: Field[Any]) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


def _encoded_equal(default: Any, value: Any) -> bool:
    """Compare a config-file value against a dataclass default through draccus encoding plus
    a JSON round-trip, so enums/paths/tuples match their YAML/JSON spellings."""
    try:
        encoded = draccus.encode(default)
    except Exception:
        return bool(default == value)
    with contextlib.suppress(TypeError, ValueError):
        encoded = json.loads(json.dumps(encoded))
    return bool(encoded == value or default == value)


def _resolve_dataclass_target(annotation: Any, value: dict[str, Any]) -> type | None:
    """Resolve the dataclass `value` will be decoded into, unwrapping Optional/Union and
    draccus choice registries (via the dict's ``type`` key). None when unresolvable."""
    candidates = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    for candidate in candidates:
        if not isinstance(candidate, type) or not is_dataclass(candidate):
            continue
        choice = value.get(draccus.CHOICE_TYPE_KEY)
        get_choice_class = getattr(candidate, "get_choice_class", None)
        if choice is None or not callable(get_choice_class):
            return candidate
        try:
            resolved = get_choice_class(str(choice))
        except KeyError:
            return None
        return resolved if isinstance(resolved, type) and is_dataclass(resolved) else None
    return None


def _class_fields_and_hints(config_class: type | None) -> tuple[dict[str, Field[Any]] | None, dict[str, Any]]:
    if config_class is None or not is_dataclass(config_class):
        return None, {}
    field_map = {f.name: f for f in dataclass_fields(config_class)}
    try:
        hints = get_type_hints(config_class)
    except Exception:
        hints = {}
    return field_map, hints


def _nondefault_leaf_names(config_class: type | None, data: dict[str, Any], prefix: str = "") -> set[str]:
    """Dotted names of the leaves in `data` that differ from `config_class`'s declared field
    defaults. Unknown classes/fields and fields without defaults count as explicit."""
    names: set[str] = set()
    field_map, hints = _class_fields_and_hints(config_class)
    for key, value in data.items():
        if key in (PATH_KEY, draccus.CHOICE_TYPE_KEY):
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        field = field_map.get(key) if field_map is not None else None
        if field is None:
            names.add(dotted)
            continue
        if isinstance(value, dict):
            target = _resolve_dataclass_target(hints.get(key, field.type), value)
            if target is not None:
                names |= _nondefault_leaf_names(target, value, dotted)
                continue
        default = _field_default(field)
        if default is MISSING or not _encoded_equal(default, value):
            names.add(dotted)
    return names


def get_explicit_override_fields(field_name: str) -> set[str]:
    overrides = get_yaml_overrides(field_name) + (get_cli_overrides(field_name) or [])
    return _config_explicit_fields.get(field_name, set()) | get_override_field_names(overrides)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    if args is None:
        return []
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = [] if args is None else list(args)

    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def extract_path_fields_from_config(
    config_path: str, path_fields: list[str], config_class: type | None = None
) -> str:
    """Extract `path` fields from a YAML/JSON config before draccus processes it.

    When a user specifies e.g. ``policy.path: lerobot/smolvla_base`` in a YAML config,
    draccus will fail because ``path`` is not a valid field on policy config classes.
    This function extracts those path values, stores them in ``_config_path_args`` for
    later retrieval by ``get_path_arg()``, and returns a cleaned temp config file path.

    ``config_class`` (the top-level config dataclass) lets non-path fields register as
    explicit overrides only where they differ from the class's declared defaults.
    """
    # Defensive reset for direct callers. wrap() already clears all three registries at the
    # start of every CLI parse -- which is what protects a second parse in the same process,
    # since this function only runs when a --config_path is passed.
    _clear_config_overrides()

    config_file = Path(config_path)
    suffix = config_file.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_file) as f:
            config_data = json.load(f)
    else:
        return config_path

    if not isinstance(config_data, dict):
        return config_path

    # Record which config-file fields the user actually chose -- those that differ from the
    # config class's declared defaults -- not just the ``path_fields`` handled below.
    # Consumers such as ``EnvConfig.reconcile_policy_contract`` ask
    # ``get_explicit_override_fields`` whether the user set a field on purpose: values
    # written in a --config_path YAML must not look unset (they'd be silently replaced by
    # checkpoint-owned values instead of raising on conflict), while a config file dumped by
    # a previous run (draccus serializes every default) must not make each field look
    # deliberately set. Comparing against defaults also covers list- and None-valued fields,
    # which CLI-style flattening silently dropped.
    field_map, hints = _class_fields_and_hints(config_class)
    for field, value in config_data.items():
        if field in path_fields:
            continue
        class_field = field_map.get(field) if field_map is not None else None
        names: set[str]
        if isinstance(value, dict):
            annotation = hints.get(field, class_field.type) if class_field is not None else None
            target = _resolve_dataclass_target(annotation, value) if class_field is not None else None
            names = _nondefault_leaf_names(target, value)
        elif class_field is None:
            names = {field}
        else:
            default = _field_default(class_field)
            names = set() if default is not MISSING and _encoded_equal(default, value) else {field}
        names |= {name.split(".", 1)[0] for name in names}
        if names:
            _config_explicit_fields[field] = names

    modified = False
    for field in path_fields:
        if field in config_data and isinstance(config_data[field], dict) and PATH_KEY in config_data[field]:
            _config_path_args[field] = str(config_data[field].pop(PATH_KEY))
            remaining = config_data[field]
            if remaining:
                _config_yaml_overrides[field] = _flatten_to_cli_args(remaining)
            del config_data[field]
            modified = True

    if not modified:
        return config_path

    # Write cleaned config to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        if suffix in (".yaml", ".yml"):
            yaml.dump(config_data, tmp, default_flow_style=False)
        else:
            json.dump(config_data, tmp, indent=2)
    return tmp.name


def wrap(config_path: Path | None = None) -> Callable[[F], F]:
    """
    HACK: Similar to draccus.wrap but does three additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments. These plugins will typically register
            their own subclasses of config classes, so that draccus can find the right class to instantiate
            from the CLI '.type' arguments
    """

    def wrapper_outer(fn: F) -> F:
        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            # The registries describe exactly one parse and are module-global. Clear
            # them for EVERY invocation -- including the pre-supplied-cfg branch below,
            # which performs no parse: a programmatically built config must not inherit
            # a previous CLI parse's explicit-override entries (a stale env.fps override
            # from an earlier --config_path run would otherwise hard-fail
            # reconcile_policy_contract on a config that never mentioned fps).
            _clear_config_overrides()
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                # (Registries were already cleared unconditionally above, before anything
                # consults them -- filter_path_args below reads the get_path_arg fallback.)
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                cli_args_before_path_filter = tuple(cli_args)
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                    # Also extract path fields from the YAML/JSON config file
                    if config_path_cli:
                        config_path_cli = extract_path_fields_from_config(
                            config_path_cli, path_fields, argtype
                        )
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    pretrained_kwargs: dict[str, Any] = {"cli_args": cli_args}
                    # Explicit opt-in only; legacy/plugin loaders retain their existing call.
                    try:
                        context_parameter = inspect.signature(argtype.from_pretrained).parameters.get(
                            "cli_args_before_path_filter"
                        )
                    except (TypeError, ValueError):
                        context_parameter = None
                    if context_parameter is not None and context_parameter.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        pretrained_kwargs["cli_args_before_path_filter"] = cli_args_before_path_filter
                    cfg = argtype.from_pretrained(config_path_cli, **pretrained_kwargs)
                else:
                    if config_path_cli:
                        cli_args = filter_arg("config_path", cli_args)
                    cfg = draccus.parse(
                        config_class=argtype,
                        config_path=config_path_cli or config_path,
                        args=cli_args,
                    )
            response = fn(cfg, *args, **kwargs)
            return response

        return cast(F, wrapper_inner)

    return cast(Callable[[F], F], wrapper_outer)
