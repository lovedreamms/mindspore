# import mindspore
# from mindspore import nn 
# from models import commen
# import numpy as np
# def Pre_train(opt):
#     # ----------------------------------------
#     #       Network training parameters
#     # ----------------------------------------

#     save_folder = opt.save_path
#     sample_folder = opt.sample_path
#     # utils.check_path(save_folder)
#     # utils.check_path(sample_folder)
#     criterion_L1 = nn.L1Loss()
#     criterion_L2 = nn.MSELoss()
#     #criterion_rainypred = mindspore.nn.L1Loss().cuda()
#     criterion_ssim = commen.SSIM()
#     generator = utils.create_generator(opt)
# 设计随机种子
# mindspore.set_seed(1)
# from mindspore.common.initializer import *
# mindspore.set_seed(1)
# net = nn.Conv1d(120, 240, 4, has_bias=False)
# print(net.__class__.__name__.find('Conv'))
# print(hasattr(net, 'weight'))
# net.weight.set_data(initializer(One(), net.weight.shape,net.weight.dtype))
# x = mindspore.Tensor(np.random.randn(1, 120, 32), mindspore.float32)
# y = net(x)


