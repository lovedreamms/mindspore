from mindspore import Tensor
tensor1 = Tensor(1)
print(type(tensor1))


import numpy as np
tensor2 = Tensor(np.array([1, 2, 3]))
print(type(tensor2))


from mindspore.common import initializer as init 
from mindspore import dtype 
tensor3 = Tensor(shape=(2, 3), dtype=dtype.float32, init=init.One())
print(type(tensor3))


from mindspore import ops
tensor4 = ops.OnesLike()(tensor2)
print((tensor4))


