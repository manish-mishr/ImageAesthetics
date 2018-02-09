import random
import socket
import datetime

import pycrayon
import numpy as np

from collections import namedtuple

'''code-block::

    expt = get_experiment(name='s2s_lr_0.1')
    # Loss as scalar
    for t in reversed(xrange(1000)):
        expt.add_scalar_dict(
            {
                'loss_mxe/train': t+random.random()*t//5,
                'loss_mxe/dev': 2*t*random.random()+random.random()*t//5,
            },
        )
        time.sleep(0.01)
    # Weights as hist
    expt.add_histogram_value(
        name='affine_1_weights',
        hist=np.random.randn(10000,).tolist(),
        tobuild=True,
    ) '''



CrayonSettings = namedtuple('CrayonSettings', ['host', 'port'])
CRAYON_SETTINGS = CrayonSettings(host='localhost', port='9119')

# Connect to the server
def get_experiment(name, settings=CRAYON_SETTINGS):
    """Creates a pycrayon experiment object to log data to."""
    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
    experiment_name = '{dt}_{host};{name}'.format(
        dt=experiment_date,
        host=socket.gethostname(),
        name=name,
    )
    return get_crayon_client(settings=settings).create_experiment(experiment_name)


def get_crayon_client(settings=CRAYON_SETTINGS):
    return pycrayon.CrayonClient(hostname=settings.host, port=settings.port)


def clear_expts(settings=CRAYON_SETTINGS):
    get_crayon_client(settings=settings).remove_all_experiments()