import sys
import numpy


def main(job_id):
    ######################
    # Model construction #
    ######################

    from theano import tensor

    from blocks.bricks import Rectifier, MLP, Softmax
    from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
                                    Flattener)
    from blocks.initialization import Uniform, Constant

    x = tensor.tensor4('images')
    y = tensor.lmatrix('targets')

    # Convolutional layers
    filter_sizes = [(5, 5)] * 3 + [(4, 4)] * 3
    num_filters = [16, 32, 64, 128, 256, 512]
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
    mlp = MLP(activations=[Rectifier(), Rectifier(), None],
              dims=[512, 256, 256, 2], weights_init=Uniform(0, 0.2),
              biases_init=Constant(0.))
    mlp.initialize()
    y_hat = mlp.apply(features)

    # Numerically stable softmax
    cost = Softmax().categorical_cross_entropy(y.flatten(), y_hat)
    cost.name = 'cost'

    ############
    # Training #
    ############

    from blocks.main_loop import MainLoop
    from blocks.algorithms import GradientDescent, Momentum
    from blocks.extensions import Printing, Timing, FinishAfter
    from blocks.extensions.saveload import Checkpoint
    from blocks.extensions.monitoring import DataStreamMonitoring
    from blocks.extensions.training import TrackTheBest
    from blocks.extensions.predicates import OnLogRecord
    from blocks.graph import ComputationGraph, apply_dropout
    from blocks.roles import INPUT
    from blocks.bricks import Linear
    from blocks.filter import VariableFilter

    from fuel.streams import ServerDataStream

    momentum = numpy.random.rand() * 0.4 + 0.5  # 0.5 - 0.9
    dropout_prob = numpy.random.rand() * 0.3 + 0.3  # 0.3 - 0.6
    learning_rate = 10 ** (numpy.random.rand() - 3)  # 10 ** -2 - 10 ** -3

    for param, value in zip(['Momentum', 'Dropout', 'Learning rate'],
                            [momentum, dropout_prob, learning_rate]):
        print('{}: {}'.format(param, value))

    training_stream = ServerDataStream(('images', 'targets'), port=job_id)
    valid_stream = ServerDataStream(('images', 'targets'), port=job_id + 50)

    cg = ComputationGraph([cost])
    dropout_cg = apply_dropout(
        cg, VariableFilter(roles=[INPUT], bricks=[Linear])(cg.variables),
        dropout_prob
    )
    algorithm = GradientDescent(cost=dropout_cg.outputs[0],
                                params=cg.parameters,
                                step_rule=Momentum(learning_rate=learning_rate,
                                                   momentum=momentum))

    track_cost = TrackTheBest('valid_cost', after_epoch=True,
                              after_batch=False)

    main_loop = MainLoop(
        data_stream=training_stream, algorithm=algorithm,
        extensions=[
            Timing(),
            DataStreamMonitoring([cost], valid_stream, prefix='valid'),
            track_cost,
            Checkpoint('{}.pkl'.format(job_id), after_epoch=True)
            .add_condition('after_epoch',
                           OnLogRecord(track_cost.notification_name),
                           ('best_{}.pkl'.format(job_id),)),
            Printing(),
            FinishAfter(after_n_epochs=200),
        ]
    )
    main_loop.run()

if __name__ == "__main__":
    main(int(sys.argv[1]))
