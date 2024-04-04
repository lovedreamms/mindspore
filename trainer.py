import mindspore
import mindspore.nn as nn 
import numpy as np
import os

from models import commen
from utils.generator import *
import dataset
def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    save_folder = opt.save_path
    sample_folder = opt.sample_path
    # utils.check_path(save_folder)
    # utils.check_path(sample_folder)
    criterion_L1 = nn.L1Loss()
    criterion_L2 = nn.MSELoss()
    #criterion_rainypred = mindspore.nn.L1Loss().cuda()
    criterion_ssim = commen.SSIM()
    generator = create_generator(opt)
    optimizer_G = nn.AdamAdam(params=generator.trainable_params(), learning_rate=opt.lr_g, beta1=opt.b1, beta2=opt.b2, weight_decay=opt.weight_decay)
    print("pretrained models loaded")
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        """
        if opt.save_mode == 'epoch':
            model_name = 'KPN_single_image_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.train_batch_size, opt.mu, opt.sigma)
        if opt.save_mode == 'iter':
            model_name = 'KPN_single_image_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.train_batch_size, opt.mu, opt.sigma)
        """

        if opt.save_mode == 'epoch':
            model_name = 'KPN_rainy_image_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'KPN_rainy_image_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    mindspore.save_checkpoint(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    mindspore.save_checkpoint(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    mindspore.save_checkpoint(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    mindspore.save_checkpoint(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    trainset = dataset.DenoisingDataset(opt)
    
# # 设计随机种子
# from mindspore.common.initializer import *
# mindspore.set_seed(1)
# net = nn.Conv1d(120, 240, 4, has_bias=False)
# print(net.__class__.__name__.find('Conv'))
# print(hasattr(net, 'weight'))
# net.weight.set_data(initializer(One(), net.weight.shape,net.weight.dtype))
# x = mindspore.Tensor(np.random.randn(1, 120, 32), mindspore.float32)
# y = net(x)
# optimizer_G = nn.Adam(params=net.trainable_params(), learning_rate=0.01, beta1=0.5, beta2=0.99, weight_decay=0)

