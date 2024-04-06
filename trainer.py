import datetime
import time
import cv2
import mindspore
import mindspore.nn as nn 
import mindspore.dataset as ds
import numpy as np
import os

from models import commen
from utils.generator import *
import dataset

def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255, height = -1, width = -1):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization
        img = img * 255.0
        # Process img_copy and do not destroy the data of img
        #print(img.size())
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        if (height != -1) and (width != -1):
            img_copy = cv2.resize(img_copy, (width, height))
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
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
    optimizer_G = nn.Adam(params=generator.trainable_params(), learning_rate=opt.lr_g, beta1=opt.b1, beta2=opt.b2, weight_decay=opt.weight_decay)
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
    print('The overall number of training images:', len(trainset))

    train_loader = ds.GeneratorDataset(trainset,shuffle = True, num_parallel_workers = opt.num_workers)
    train_loader.batch(opt.train_batch_size)

     # ----------------------------------------
    #                 Training
    # ----------------------------------------
    prev_time = time.time()
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(train_loader):

            print("in epoch %d" % i)
            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input, true_input)
            
            ssim_loss = -criterion_ssim(true_target, fake_target)
            '''
            #trans for enc_net
            enc_trans = transforms.Compose([transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            fake_target_norm = torch.from_numpy(np.zeros(fake_target.size())).cuda()
            true_target_norm = torch.from_numpy(np.zeros(true_target.size())).cuda()
            for j in range(fake_target.size()[0]):
                fake_target_norm[j] = enc_trans(fake_target[j])
                true_target_norm[j] = enc_trans(true_target[j])
            ''' 

            #print(fake_target_norm.size())
            #enc_pred = encnet.evaluate(fake_target_norm.type(torch.FloatTensor).cuda())
            #enc_pred = encnet(fake_target_norm.type(torch.FloatTensor).cuda())[0]
            #enc_gt = encnet(true_target_norm.type(torch.FloatTensor).cuda())[0]

            '''
            enc_feat_pred = encnet_feat(fake_target_norm.type(torch.FloatTensor).cuda())[0]
            enc_feat_gt = encnet_feat(true_target_norm.type(torch.FloatTensor).cuda())[0]
            '''

            #rain_layer_gt = true_input - true_target 
            #rain_layer_pred = true_input - fake_target
            #rainy_pred = true_input - (fake_target * rain_layer_pred) 
            #print(type(true_input))
            #print(type(fake_target))

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            #enc_loss = criterion_L1(enc_pred, enc_gt)
            #enc_feat_loss = criterion_L1(enc_feat_pred, enc_feat_gt)
            #Pixellevel_L2_Loss = criterion_L2(fake_target, true_target)
            #Pixellevel_L2_Loss = criterion_L2(rain_layer_pred, rain_layer_gt)
            #Loss_rainypred = criterion_rainypred(rainy_pred, true_input)
            
            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + 0.2*ssim_loss
            #loss = Pixellevel_L1_Loss
            #loss = Pixellevel_L1_Loss + Pixellevel_L2_Loss + Loss_rainypred
            loss.backward()
            optimizer_G.step()

            #check
            '''
            for j in encnet.named_parameters():
                print(j)
                break
            '''

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.4f %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), ssim_loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
       
        """### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))"""
        
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

