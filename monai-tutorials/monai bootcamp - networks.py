# title: monai bootcamp - networks
# author: Sun Yeop Lee

'''
Overview
This notebook introduces you to the MONAI network APIs:

Convolutions
Specifying layers with additional arguments
Flexible definitions of networks
'''

import torch
import monai
monai.config.print_config()
from monai.networks.layers import Conv
from monai.networks.layers import Act
from monai.networks.layers import split_args
from monai.networks.layers import Pool

# -- Unifying the network layer APIs
'''
Network functionality represents a major design opportunity for MONAI. 
Pytorch is very much unopinionated in how networks are defined. 
It provides Module as a base class from which to create a network, 
and a few methods that must be implemented, but there is no prescribed pattern nor much helper functionality for initializing networks.

This leaves a lot of room for defining some useful 'best practice' patterns for constructing new networks in MONAI. 
Although trivial, inflexible network implementations are easy enough, we can give users a toolset that makes it much easier to build well-engineered, flexible networks, and demonstrate their value by committing to use them in the networks that we build.
'''

## convolution as an example
print(Conv.__doc__)
'''
The Conv class has two options for the first argument. The second argument must be the number of spatial dimensions, Conv[name, dimension]
'''

print(Conv[Conv.CONV, 1])
print(Conv[Conv.CONV, 2])
print(Conv[Conv.CONV, 3])
print(Conv[Conv.CONVTRANS, 1])
print(Conv[Conv.CONVTRANS, 2])
print(Conv[Conv.CONVTRANS, 3])

'''
The configured classes are the "vanilla" PyTorch layers. 
We could create instances of them by specifying the layer arguments:
'''

print(Conv[Conv.CONV, 2](in_channels=1, out_channels=4, kernel_size=3))
print(Conv[Conv.CONV, 3](in_channels=1, out_channels=4, kernel_size=3))

## Specifying a layer with additional arguments
print(Act.__doc__)

print(Act[Act.PRELU])
Act[Act.PRELU](num_parameters=1, init=0.1)

act_name, act_args = split_args(("prelu", {"num_parameters": 1, "init": 0.1}))
Act[act_name](**act_args)

## putting them together

class MyNetwork(torch.nn.Module):

  def __init__(self, dims=3, in_channels=1, out_channels=8, kernel_size=3, pool_kernel=2, act="relu"):
    super(MyNetwork, self).__init__()
    # convolution
    self.conv = Conv[Conv.CONV, dims](in_channels, out_channels, kernel_size=kernel_size)
    # activation
    act_type, act_args = split_args(act)
    self.act = Act[act_type](**act_args)
    # pooling
    self.pool = Pool[Pool.MAX, dims](pool_kernel)
  
  def forward(self, x: torch.Tensor):
    x = self.conv(x)
    x = self.act(x)
    x = self.pool(x)
    return x

# default network instance
default_net = MyNetwork()
print(default_net)
print(default_net(torch.ones(3, 1, 20, 20, 30)).shape)

# 2D network instance
elu_net = MyNetwork(dims=2, in_channels=3, act=("elu", {"inplace": True}))
print(elu_net)
print(elu_net(torch.ones(3, 3, 24, 24)).shape)

# 3D network instance with anisotropic kernels
sigmoid_net = MyNetwork(3, in_channels=4, kernel_size=(3, 3, 1), act="sigmoid")
print(sigmoid_net)
print(sigmoid_net(torch.ones(3, 4, 30, 30, 5)).shape)

'''
Almost all the MONAI layers, blocks and networks are extensions of torch.nn.modules and follow this pattern. 
This makes the implementations compatible with any PyTorch pipelines and flexible with the network design. 
The current collections of those differentiable modules are listed in https://docs.monai.io/en/latest/networks.html.
'''



























