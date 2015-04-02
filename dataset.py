import os

import h5py
import numpy

from fuel import config
from fuel.datasets import IndexableDataset
from fuel.utils import do_not_pickle_attributes


@do_not_pickle_attributes('f')
class DogsVsCats(IndexableDataset):
    provides_sources = ('images', 'targets')

    def __init__(self, which_set):
        if which_set == 'train':
            self.start = 0
            self.stop = 22500
        elif which_set == 'test':
            self.start = 22500
            self.stop = 25000
        else:
            raise ValueError
        self.load()

    def load(self):
        if os.path.exists('dogs_vs_cats.hdf5'):
            self.f = h5py.File('dogs_vs_cats.hdf5')
        else:
            self.f = h5py.File(os.path.join(config.data_path,
                                            'dogs_vs_cats.hdf5'))

    @property
    def num_examples(self):
        return self.stop - self.start

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError
        images, targets = [], []
        indices, reverse = numpy.unique(request, return_inverse=True)
        indices = list(indices)
        for image, shape, target in zip(self.f['images'][indices],
                                        self.f['shapes'][indices],
                                        self.f['labels'][indices]):
            images.append(image.reshape(shape))
            targets.append([target])
        images = numpy.asarray(images)[reverse]
        targets = numpy.asarray(targets)[reverse]
        return self.filter_sources((images, targets))
