######################
# Model construction #
######################

from theano import tensor

from blocks.bricks import Rectifier, MLP  # , Softmax
# from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
                                Flattener)
from blocks.initialization import Uniform, Constant

x = tensor.tensor4('images')
y = tensor.lmatrix('targets')

# Convolutional layers

filter_sizes = [(5, 5)] * 3 + [(4, 4)] * 3
num_filters = [32, 32, 64, 64, 128, 256]
pooling_sizes = [(2, 2)] * 6
activation = Rectifier().apply
conv_layers = [
    ConvolutionalLayer(activation, filter_size, num_filters_, pooling_size)
    for filter_size, num_filters_, pooling_size
    in zip(filter_sizes, num_filters, pooling_sizes)
]
convnet = ConvolutionalSequence(conv_layers, num_channels=3,
                                image_size=(260, 260),
                                weights_init=Uniform(0, 0.2),
                                biases_init=Constant(0.))
convnet.initialize()

# Fully connected layers

features = Flattener().apply(convnet.apply(x))
mlp = MLP(activations=[Rectifier(), None],
          dims=[256, 256, 2], weights_init=Uniform(0, 0.2),
          biases_init=Constant(0.))
mlp.initialize()
y_hat = mlp.apply(features)

# Numerically stable softmax

# cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
z = y_hat - y_hat.max(axis=1).dimshuffle(0, 'x')
log_prob = z - tensor.log(tensor.exp(z).sum(axis=1).dimshuffle(0, 'x'))
flat_log_prob = log_prob.flatten()
range_ = tensor.arange(y.shape[0])
flat_indices = y.flatten() + range_ * 2
log_prob_of = flat_log_prob[flat_indices].reshape(y.shape, ndim=2)
cost = -log_prob_of.mean()
cost.name = 'cost'

# Print sizes to check
print("Representation sizes:")
for layer in convnet.layers:
    print(layer.get_dim('input_'))
    print(layer.get_dim('output'))

############
# Training #
############

from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Momentum
from blocks.extensions import Printing, SimpleExtension
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph

from dataset import DogsVsCats
from streams import RandomPatch
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

training_stream = DataStream(DogsVsCats('train'),
                             iteration_scheme=ShuffledScheme(20000, 32))
training_stream = RandomPatch(training_stream, 270, (260, 260))

cg = ComputationGraph([cost])
algorithm = GradientDescent(cost=cost, params=cg.parameters, step_rule=Momentum(learning_rate=0.001,
                                                          momentum=0.1))

main_loop = MainLoop(
    data_stream=training_stream, algorithm=algorithm,
    extensions=[
        DataStreamMonitoring(
            [cost],
            RandomPatch(DataStream(
                DogsVsCats('valid'),
                iteration_scheme=SequentialScheme(2500, 32)),
                270, (260, 260)),
            prefix='valid'
        ),
        Printing(),
        Checkpoint('dogs_vs_cats.pkl', after_epoch=True),
    ]
)
main_loop.run()
