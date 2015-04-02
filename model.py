import sys


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
    from blocks.graph import ComputationGraph

    from fuel.streams import ServerDataStream

    training_stream = ServerDataStream(('images', 'targets'), port=job_id)
    valid_stream = ServerDataStream(('images', 'targets'), port=job_id + 50)

    cg = ComputationGraph([cost])
    algorithm = GradientDescent(cost=cost, params=cg.parameters,
                                step_rule=Momentum(learning_rate=0.001,
                                                   momentum=0.9))

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
            FinishAfter(after_n_epochs=150),
        ]
    )
    main_loop.run()

if __name__ == "__main__":
    main(int(sys.argv[1]))
