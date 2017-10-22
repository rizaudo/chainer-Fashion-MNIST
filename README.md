# chainer-Fashion-MNIST
Provide simple function for grab Fashion-MNIST(`https://github.com/zalandoresearch/fashion-mnist`) dataset.

## Requirements
Python 3.5+, 3.6+ (sorry, I didn't test in python2.),
Chainer, numpy

## Install
Place fashionminist.py to PYTHONPATH, 
then use `get_fmnist` like chainer's `get_mnist`.

## Example
``` python
from fashionmnist import get_fmnist
train, test = get_fmnist(withlabel=True, ndim=3, scale=255.)
# <DO SOMETHING>
```

## Question or Found BUG?
Please registering an issue.

## NOTE
Original mnist.py is part of Chainer(https://chainer.org/).

## License
This Code provided as MIT License (Please see `LICENSE` file).
