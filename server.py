import sys
import numpy
from multiprocessing import Process

from fuel.schemes import SequentialScheme
from fuel.server import start_server
from fuel.streams import DataStream

from dataset import DogsVsCats
from streams import RandomPatch


def open_stream(examples, port, p_rotate=0, p_flip=0):
    stream = DataStream(
        DogsVsCats('train'),
        iteration_scheme=SequentialScheme(examples, 64)
    )
    stream = RandomPatch(stream, 280, (260, 260),
                         p_rotate=p_rotate, p_flip=p_flip)
    start_server(stream, port=port)

if __name__ == "__main__":
    train_examples = numpy.random.choice(22500, size=20000)
    with open('{}_train_set.txt'.format(sys.argv[1]), 'w') as f:
        f.write('\n'.join(map(str, train_examples)))
    valid_examples = numpy.setdiff1d(numpy.arange(22500), train_examples)
    Process(target=open_stream, args=(train_examples,
            int(sys.argv[1]), 0.1, 0.5)).start()
    Process(target=open_stream, args=(valid_examples,
            int(sys.argv[1]) + 50)).start()
