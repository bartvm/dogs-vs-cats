import sys

from fuel.schemes import ShuffledScheme
from fuel.server import start_server
from fuel.streams import DataStream

from dataset import DogsVsCats
from streams import RandomPatch


def main(job_id):

    training_stream = DataStream(DogsVsCats('train'),
                                 iteration_scheme=ShuffledScheme(20000, 64))
    training_stream = RandomPatch(training_stream, 280, (260, 260))

    start_server(training_stream, port=job_id)

if __name__ == "__main__":
    main(sys.argv[1])
