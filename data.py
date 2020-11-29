"""Tools for loading, shuffling, and batching SANI npy datasets

The `load(path)` creates an iterable of raw data,
where species are strings, and coordinates are numpy ndarrays.

You can transform these iterable by using transformations.
To do transformation, just do `it.transformation_name()`.

Available transformations are listed below:

- `shuffle`
- `cache` cache the result of previous transformations.
- `collate` pad the dataset, convert it to tensor, and stack them
    together to get a batch.

Example:

.. code-block:: python

    training = torchani.data.load(dspath).shuffle()
    training = training.collate(batch_size).cache()


.. code-block:: python

    training = torchani.data.load(dspath).shuffle()
    training = torch.utils.data.DataLoader(list(training), batch_size=batch_size, collate_fn=data.collate_fn, num_workers=64)

"""

import gc
import random
import torch
import glob
import importlib
import functools
import numpy as np
from collections import defaultdict

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True


PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'energies': 0.0,
    'charges': 0.0
}


def empty_list():
    return []


def stack_with_padding(properties, padding):
    output = defaultdict(empty_list)
    for p in properties:
        for k, v in p.items():
            output[k].append(torch.as_tensor(v))
    for k, v in output.items():
        if v[0].dim() == 0:
            output[k] = torch.stack(v)
        else:
            output[k] = torch.nn.utils.rnn.pad_sequence(v, True, padding[k])
    return output


def collate_fn(samples):
    return stack_with_padding(samples, PADDING)


class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())


class IterableAdapterWithLength(IterableAdapter):

    def __init__(self, iterable_factory, length):
        super().__init__(iterable_factory)
        self.length = length

    def __len__(self):
        return self.length


class Transformations:
    """Convert one reenterable iterable to another reenterable iterable"""

    @staticmethod
    def shuffle(reenterable_iterable):
        list_ = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(reenterable_iterable):
        ret = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        return ret

    @staticmethod
    def collate(reenterable_iterable, batch_size):
        def reenterable_iterable_factory():
            batch = []
            i = 0
            for d in reenterable_iterable:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield collate_fn(batch)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch)
        try:
            length = (len(reenterable_iterable) + batch_size - 1) // batch_size
            return IterableAdapterWithLength(reenterable_iterable_factory, length)
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def pin_memory(reenterable_iterable):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                yield {k: d[k].pin_memory() for k in d}
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def remove_outliers(reenterable_iterable, stats):
        def custom_filter():
            for d in reenterable_iterable:
                net_charge = np.rint(d['charges'].sum())
                mean, std = stats[net_charge]

                upper = mean + 7 * std * min(1/abs(net_charge), 1)
                lower = mean - 5 * std * min(1/abs(net_charge), 1)
                if d['energies'] > lower and d['energies'] < upper:
                    yield d

        return IterableAdapter(custom_filter)


class TransformableIterable:
    def __init__(self, wrapped_iterable, transformations=()):
        self.wrapped_iterable = wrapped_iterable
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iterable)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self.wrapped_iterable, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        length = len(self)
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(int(n * length)):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        del self_iter
        gc.collect()
        return iters

    def __len__(self):
        return len(self.wrapped_iterable)


def anidata_loader(path, additional_properties=()):

    def get_data(path):
        data_paths = sorted(glob.glob(path))
        use_pbar = PKBAR_INSTALLED and verbose
        if use_pbar:
            pbar = pkbar.Pbar(f" loading {path}", len(data_paths))

        for i, data_dir in enumerate(data_paths):
            d = dict()

            xs = np.load(data_dir + '/xs.npy').reshape(-1, 4)
            n_atoms = np.load(data_dir + '/atom_counts.npy')
            n_atoms_cumsum = np.cumsum(n_atoms, dtype=np.int)

            d['xs'] = np.split(xs, n_atoms_cumsum[:-1])
            d['ys'] = np.load(data_dir + '/ys.npy')

            for prop in additional_properties:
                prop_data = np.load(data_dir + f'/{prop}.npy')

                # atomic properties
                if 'charges' in prop:  # unique key for diff charge methods
                    prop = 'charges'
                d[prop] = np.split(prop_data, n_atoms_cumsum[:-1])

            yield d

            if use_pbar:
                pbar.update(i)

    def conformations():
        for d in get_data(path):
            for i, species_coordinates in enumerate(d['xs']):
                species = species_coordinates[:, 0]
                coordinates = species_coordinates[:, 1:]
                energies = d['ys'][i]
                ret = {
                    'species': species.astype('int64'),
                    'coordinates': coordinates,
                    'energies': energies
                }
                for prop in additional_properties:
                    if 'charges' in prop:
                        prop = 'charges'
                    ret[prop] = d[prop][i].astype('float32')
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))


# if __name__ == "__main__":
    # training = load('data/*train*').shuffle()
    # print(len(training))
