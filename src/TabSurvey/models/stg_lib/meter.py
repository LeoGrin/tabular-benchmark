import six
import itertools
import collections
import json

def map_exec(func, *iterables):
    return list(map(func, *iterables))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    val = 0
    avg = 0
    sum = 0
    count = 0
    tot_count = 0

    def __init__(self):
        self.reset()
        self.tot_count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.tot_count += n
        self.avg = self.sum / self.count


class GroupMeters(object):
    def __init__(self):
        self._meters = collections.defaultdict(AverageMeter)

    def reset(self):
        map_exec(AverageMeter.reset, self._meters.values())

    def update(self, updates=None, value=None, n=1, **kwargs):
        """
        Example:
            >>> meters.update(key, value)
            >>> meters.update({key1: value1, key2: value2})
            >>> meters.update(key1=value1, key2=value2)
        """
        if updates is None:
            updates = {}
        if updates is not None and value is not None:
            updates = {updates: value}
        updates.update(kwargs)
        for k, v in updates.items():
            self._meters[k].update(v, n=n)

    def __getitem__(self, name):
        return self._meters[name]

    def items(self):
        return self._meters.items()

    @property
    def sum(self):
        return {k: m.sum for k, m in self._meters.items() if m.count > 0}

    @property
    def avg(self):
        return {k: m.avg for k, m in self._meters.items() if m.count > 0}

    @property
    def val(self):
        return {k: m.val for k, m in self._meters.items() if m.count > 0}

    def format(self, caption, values, kv_format, glue):
        meters_kv = self._canonize_values(values)
        log_str = [caption]
        log_str.extend(itertools.starmap(kv_format.format, sorted(meters_kv.items())))
        return glue.join(log_str)

    def format_simple(self, caption, values='avg', compressed=True):
        if compressed:
            return self.format(caption, values, '{}={:4f}', ' ')
        else:
            return self.format(caption, values, '\t{} = {:4f}', '\n')

    def dump(self, filename, values='avg'):
        meters_kv = self._canonize_values(values)
        with open(filename, 'a') as f:
            #f.write(io.dumps_json(meters_kv, compressed=False))
            f.write(json.dumps(meters_kv, cls=JsonObjectEncoder, sort_keys=True, indent=4, separators=(',', ': ')))
            f.write('\n')

    def _canonize_values(self, values):
        if isinstance(values, six.string_types):
            assert values in ('avg', 'val', 'sum')
            meters_kv = getattr(self, values)
        else:
            meters_kv = values
        return meters_kv


class JsonObjectEncoder(json.JSONEncoder):
    """Adapted from https://stackoverflow.com/a/35483750"""

    def default(self, obj):
        if hasattr(obj, '__jsonify__'):
            json_object = obj.__jsonify__()
            if isinstance(json_object, six.string_types):
                return json_object
            return self.encode(json_object)
        else:
            raise TypeError("Object of type '%s' is not JSON serializable." % obj.__class__.__name__)

        if hasattr(obj, '__dict__'):
            d = dict(
                (key, value)
                for key, value in inspect.getmembers(obj)
                if not key.startswith("__")
                and not inspect.isabstract(value)
                and not inspect.isbuiltin(value)
                and not inspect.isfunction(value)
                and not inspect.isgenerator(value)
                and not inspect.isgeneratorfunction(value)
                and not inspect.ismethod(value)
                and not inspect.ismethoddescriptor(value)
                and not inspect.isroutine(value)
            )
            return self.default(d)

        return obj