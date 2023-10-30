# This file must not import anything from lib except for lib/env.py

import argparse
import dataclasses
import datetime
import enum
import importlib
import inspect
import json
import os
import pickle
import shutil
import sys
import time
import typing
import warnings
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Optional, Type, TypeVar, Union

import delu
import numpy as np
import tomli
import tomli_w
import torch
from loguru import logger

from . import env


def configure_libraries():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.allow_tf32 = False  # type: ignore[code]
    torch.backends.cudnn.benchmark = False  # type: ignore[code]
    torch.backends.cudnn.deterministic = True  # type: ignore[code]

    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


# ======================================================================================
# >>> types <<<
# ======================================================================================
KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # must be JSON-serializable
T = TypeVar('T')


# ======================================================================================
# >>> enums <<<
# ======================================================================================
class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


# ======================================================================================
# >>> IO <<<
# ======================================================================================
def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


# ======================================================================================
# >>> Function <<<
# ======================================================================================
# "Function" is any function with the following signature:
# Function: (
#     config: JSONDict,
#     output: Union[str, Path],
#     *,
#     force = False,
#     [continue_ = False]
# ) -> Optional[JSONDict]
Function = Callable[..., Optional[JSONDict]]


def start(
    output: Union[str, Path], force: bool = False, continue_: bool = False
) -> bool:
    """Start Function."""
    print_sep('=')
    output = env.get_path(output)
    print(
        f'[>>>] {env.try_get_relative_path(output)} | {datetime.datetime.now()}'  # noqa: E501
    )

    if output.exists():
        if force:
            logger.warning('Removing the existing output')
            time.sleep(1.0)
            shutil.rmtree(output)
            output.mkdir()
            return True
        elif not continue_:
            backup_output(output)
            logger.warning('Already exists!')
            return False
        elif output.joinpath('DONE').exists():
            backup_output(output)
            logger.info('Already done!\n')
            return False
        else:
            logger.info('Continuing with the existing output')
            return True
    else:
        logger.info('Creating the output')
        output.mkdir()
        return True


def make_config(Config: Type[T], config: JSONDict) -> T:
    assert is_dataclass(Config) or Config is dict

    if isinstance(config, Config):
        the_config = config
    else:
        assert is_dataclass(Config)

        def _from_dict(datacls: type[T], data: dict) -> T:
            # this is an intentionally restricted parsing which
            # supports only nested (optional) dataclasses,
            # but not unions and collections thereof
            assert is_dataclass(datacls)
            data = deepcopy(data)
            for field in dataclasses.fields(datacls):
                if field.name not in data:
                    continue
                if is_dataclass(field.type):
                    data[field.name] = _from_dict(field.type, data[field.name])
                # check if Optional[<dataclass>]
                elif (
                    typing.get_origin(field.type) is Union
                    and len(typing.get_args(field.type)) == 2
                    and typing.get_args(field.type)[1] is type(None)
                    and is_dataclass(typing.get_args(field.type)[0])
                ):
                    if data[field.name] is not None:
                        data[field.name] = _from_dict(
                            typing.get_args(field.type)[0], data[field.name]
                        )
                else:
                    # in this case, we do nothing and hope for good luck
                    pass

            return datacls(**data)

        the_config = _from_dict(Config, config)

    print_sep()
    pprint(
        asdict(the_config) if is_dataclass(the_config) else the_config,
        sort_dicts=False,
        width=100,
    )
    print_sep()
    return the_config


def create_report(config: JSONDict) -> JSONDict:
    # report is just a JSON-serializable Python dictionary
    # for storing arbitrary information about a given run.

    report = {}
    # If this snippet succeeds, then report['function'] will the full name of the
    # function relative to the project directory (e.g. "bin.xgboost_.main")
    try:
        caller_frame = inspect.stack()[1]
        caller_relative_path = env.get_path(caller_frame.filename).relative_to(
            env.PROJECT_DIR
        )
        caller_module = str(caller_relative_path.with_suffix('')).replace('/', '.')
        caller_function_qualname = f'{caller_module}.{caller_frame.function}'
        import_(caller_function_qualname)
        report['function'] = caller_function_qualname
    except Exception as err:
        warnings.warn(
            f'The key "function" will be missing in the report. Reason: {err}'
        )
    report['gpus'] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

    def jsonify(value):
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return value
        elif isinstance(value, list):
            return [jsonify(x) for x in value]
        elif isinstance(value, dict):
            return {k: jsonify(v) for k, v in value.items()}
        else:
            return '<nonserializable>'

    report['config'] = jsonify(config)
    return report


def summarize(report: JSONDict) -> JSONDict:
    summary = {'function': report.get('function')}

    if 'best' in report:
        # The gpus info is collected from the best report.
        summary['best'] = summarize(report['best'])
    else:
        gpus = report.get('gpus')
        if gpus is None:
            env = report.get('environment')
            summary['gpus'] = (
                []
                if env is None
                else [
                    env['gpus']['devices'][i]['name']
                    for i in map(int, env.get('CUDA_VISIBLE_DEVICES', '').split(','))
                ]
            )
        else:
            summary['gpus'] = gpus

    for key in ['n_parameters', 'best_stage', 'best_epoch', 'tuning_time', 'trial_id']:
        if key in report:
            summary[key] = deepcopy(report[key])

    metrics = report.get('metrics')
    if metrics is not None and 'score' in next(iter(metrics.values())):
        summary['scores'] = {part: metrics[part]['score'] for part in metrics}

    for key in ['n_completed_trials', 'time']:
        if key in report:
            summary[key] = deepcopy(report[key])

    return summary


def run_Function_cli(function: Function, argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('--force', action='store_true')
    if 'continue_' in inspect.signature(function).parameters:
        can_continue_ = True
        parser.add_argument('--continue', action='store_true', dest='continue_')
    else:
        can_continue_ = False
    args = parser.parse_args(*(() if argv is None else (argv,)))

    # >>> snippet for the internal infrastructure, ignore it
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert can_continue_ and args.continue_
    # <<<

    config_path = env.get_path(args.config)
    assert config_path.exists()
    function(
        load_config(config_path),
        config_path.with_suffix(''),
        force=args.force,
        **({'continue_': args.continue_} if can_continue_ else {}),
    )


_LAST_SNAPSHOT_TIME = None


def backup_output(output: Path) -> None:
    """
    This is a function for the internal infrastructure, ignore it.
    """
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output.relative_to(env.PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output = dir_ / relative_output_dir
        prev_backup_output = new_output.with_name(new_output.name + '_prev')
        new_output.parent.mkdir(exist_ok=True, parents=True)
        if new_output.exists():
            new_output.rename(prev_backup_output)
        shutil.copytree(output, new_output)
        # the case for evaluate.py which automatically creates configs
        if output.with_suffix('.toml').exists():
            shutil.copyfile(
                output.with_suffix('.toml'), new_output.with_suffix('.toml')
            )
        if prev_backup_output.exists():
            shutil.rmtree(prev_backup_output)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot  # type: ignore[code]

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def finish(output: Path, report: JSONDict) -> None:
    dump_json(report, output / 'report.json')
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output.relative_to(env.PROJECT_DIR))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_report(output)
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output.joinpath('DONE').touch()
    backup_output(output)
    print()
    print_sep()
    try:
        print_summary(output)
    except FileNotFoundError:
        pass
    print_sep()
    print(f'[<<<] {env.try_get_relative_path(output)} | {datetime.datetime.now()}')


# ======================================================================================
# >>> output <<<
# ======================================================================================
_TOML_CONFIG_NONE = '__null__'


def _process_toml_config(data, load) -> JSONDict:
    if load:
        # replace _TOML_CONFIG_NONE with None
        condition = lambda x: x == _TOML_CONFIG_NONE  # noqa
        value = None
    else:
        # replace None with _TOML_CONFIG_NONE
        condition = lambda x: x is None  # noqa
        value = _TOML_CONFIG_NONE

    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)  # type: ignore[code]


def load_config(output_or_config_path: Union[str, Path]) -> JSONDict:
    with open(env.get_path(output_or_config_path).with_suffix('.toml'), 'rb') as f:
        return _process_toml_config(tomli.load(f), True)


def dump_config(config: JSONDict, output_or_config_path: Union[str, Path]) -> None:
    path = env.get_path(output_or_config_path).with_suffix('.toml')
    with open(path, 'wb') as f:
        tomli_w.dump(_process_toml_config(config, False), f)
    assert config == load_config(path)  # sanity check


def load_report(output: Union[str, Path]) -> JSONDict:
    return load_json(env.get_path(output) / 'report.json')


def dump_report(report: JSONDict, output: Union[str, Path]) -> None:
    dump_json(report, env.get_path(output) / 'report.json')


def load_summary(output: Union[str, Path]) -> JSONDict:
    return load_json(env.get_path(output) / 'summary.json')


def print_summary(output: Union[str, Path]):
    pprint(load_summary(output), sort_dicts=False, width=60)


def dump_summary(summary: JSONDict, output: Union[str, Path]) -> None:
    dump_json(summary, env.get_path(output) / 'summary.json')


def load_predictions(output: Union[str, Path]) -> dict[str, np.ndarray]:
    x = np.load(env.get_path(output) / 'predictions.npz')
    return {key: x[key] for key in x}


def dump_predictions(
    predictions: dict[str, np.ndarray], output: Union[str, Path]
) -> None:
    np.savez(env.get_path(output) / 'predictions.npz', **predictions)


def get_checkpoint_path(output: Union[str, Path]) -> Path:
    return env.get_path(output) / 'checkpoint.pt'


def load_checkpoint(output: Union[str, Path], **kwargs) -> JSONDict:
    return torch.load(get_checkpoint_path(output), **kwargs)


def dump_checkpoint(checkpoint: JSONDict, output: Union[str, Path], **kwargs) -> None:
    torch.save(checkpoint, get_checkpoint_path(output), **kwargs)


# ======================================================================================
# >>> other <<<
# ======================================================================================
def celebrate():
    print('🌸 New best epoch! 🌸')


def print_sep(ch='-'):
    print(ch * 80)


def print_metrics(loss: float, metrics: dict) -> None:
    print(
        f'(val) {metrics["val"]["score"]:.3f}'
        f' (test) {metrics["test"]["score"]:.3f}'
        f' (loss) {loss:.5f}'
    )


def log_scores(metrics: dict) -> None:
    logger.debug(
        f'[val] {metrics["val"]["score"]:.4f} [test] {metrics["test"]["score"]:.4f}'
    )


def run_timer():
    timer = delu.Timer()
    timer.run()
    return timer


def import_(qualname: str) -> Any:
    # example: import_('bin.catboost_.main')
    try:
        module, name = qualname.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    except Exception as err:
        raise ValueError(f'Cannot import "{qualname}"') from err


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert (
            os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        ), 'Set CUDA_VISIBLE_DEVICES explicitly, e.g. `export CUDA_VISIBLE_DEVICES=0`'
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def run_cli(fn: Callable, argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    for name, arg in inspect.signature(fn).parameters.items():
        origin = typing.get_origin(arg.annotation)
        if origin is Union:
            # must be optional
            assert len(typing.get_args(arg.annotation)) == 2 and (
                typing.get_args(arg.annotation)[1] is type(None)
            )
            assert arg.default is None
            type_ = typing.get_args(arg.annotation)[0]
        else:
            assert origin is None
            type_ = arg.annotation

        has_default = arg.default is not inspect.Parameter.empty
        if arg.annotation is bool:
            if not has_default or not arg.default:
                parser.add_argument('--' + name, action='store_true')
            else:
                parser.add_argument('--no-' + name, action='store_false', dest=name)
        else:
            assert type_ in [int, float, str, bytes, Path] or issubclass(
                type_, enum.Enum
            )
            parser.add_argument(
                ('--' if has_default else '') + name,
                type=((lambda x: bytes(x, 'utf8')) if type_ is bytes else type_),
                **({'default': arg.default} if has_default else {}),
            )
    args = parser.parse_args(*(() if argv is None else (argv,)))
    return fn(**vars(args))
