import mindspore.dataset as Dataset
import cv2
import random
import numpy as np

class DenoisingDataset(Dataset):
    def __init__(self, opt):                                   		    # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        self.rainaug = opt.rainaug
        '''
        for pair in self.imglist:
            print(pair[0] + ' | ' + pair[1])
        '''
    def getRainLayer2(self, rand_id1, rand_id2):
        path_img_rainlayer_src = "./rainmix/Streaks_Garg06/" + str(rand_id1) + "-" + str(rand_id2) + ".png"
        rainlayer_rand = cv2.imread(path_img_rainlayer_src).astype(np.float32) / 255.0
        rainlayer_rand = cv2.cvtColor(rainlayer_rand, cv2.COLOR_BGR2RGB)
        return rainlayer_rand

    def getRandRainLayer2(self):
        rand_id1 = random.randint(1, 165)
        rand_id2 = random.randint(4, 8)
        rainlayer_rand = self.getRainLayer2(rand_id1, rand_id2)
        return rainlayer_rand
        
    def rain_aug(self, img_rainy, img_gt):
        img_rainy = (img_rainy.astype(np.float32)) / 255.0
        img_gt = (img_gt.astype(np.float32)) / 255.0
        if random.randint(0, 10) > 3:
            img_rainy_ret = img_rainy
        else:
            img_rainy_ret = img_gt
        img_gt_ret = img_gt

        rainlayer_rand2 = self.getRandRainLayer2()
        rainlayer_aug2 = augment_and_mix.augment_and_mix(rainlayer_rand2, severity = 3, width = 3, depth = -1) * 1
        #rainlayer_rand2ex = self.getRandRainLayer2()
        #rainlayer_aug2ex = augment_and_mix.augment_and_mix(rainlayer_rand2ex, severity = 3, width = 3, depth = -1) * 1

        height = min(img_gt.shape[0], rainlayer_aug2.shape[0])
        width = min(img_gt.shape[1], rainlayer_aug2.shape[1])
        #height = min(img_gt.shape[0], min(rainlayer_aug2.shape[0], rainlayer_aug2ex.shape[0]))
        #width = min(img_gt.shape[1], min(rainlayer_aug2.shape[1], rainlayer_aug2ex.shape[1]))
        
        cropper = RandomCrop(rainlayer_aug2.shape[:2], (height, width))
        rainlayer_aug2_crop = cropper(rainlayer_aug2)
        #cropper = RandomCrop(rainlayer_aug2ex.shape[:2], (height, width))
        #rainlayer_aug2ex_crop = cropper(rainlayer_aug2ex)
        #print(height, width, rainlayer_aug2_crop.shape, rainlayer_aug2ex_crop.shape)
        #rainlayer_aug2_crop = rainlayer_aug2_crop + rainlayer_aug2ex_crop
        cropper = RandomCrop(img_gt_ret.shape[:2], (height, width))
        img_rainy_ret = cropper(img_rainy_ret)
        img_gt_ret = cropper(img_gt_ret)
        img_rainy_ret = img_rainy_ret + rainlayer_aug2_crop - img_rainy_ret*rainlayer_aug2_crop
        np.clip(img_rainy_ret, 0.0, 1.0)
        
        img_rainy_ret = img_rainy_ret * 255
        img_gt_ret = img_gt_ret * 255
        
        #cv2.imwrite("./temp/temp.jpg", cv2.cvtColor(img_rainy_ret, cv2.COLOR_RGB2BGR))
        
        return img_rainy_ret, img_gt_ret
        
    def __getitem__(self, index):
        ## read an image
        img_rainy = cv2.imread(self.imglist[index][0])
        img_gt = cv2.imread(self.imglist[index][1])
        img_rainy = cv2.cvtColor(img_rainy, cv2.COLOR_BGR2RGB)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        if self.rainaug:
            img_rainy, img_gt = self.rain_aug(img_rainy, img_gt)
            
        '''
        if img_rainy.shape[0] < self.opt.crop_size or img_rainy.shape[1] < self.opt.crop_size:
            img_rainy=cv2.copyMakeBorder(img_rainy, 0, max(0, -img_rainy.shape[0]+self.opt.crop_size), 0, max(0, -img_rainy.shape[1]+self.opt.crop_size), cv2.BORDER_CONSTANT, value=0)
            img_gt=cv2.copyMakeBorder(img_gt, 0, max(0, -img_gt.shape[0]+self.opt.crop_size), 0, max(0, -img_gt.shape[1]+self.opt.crop_size), cv2.BORDER_CONSTANT, value=0)
            #print(img_rainy.shape[0], img_rainy.shape[1])
        '''
        '''
        ## data augmentation 这段色彩通道的位置好像不匹配吧
        # random scale
        if self.opt.geometry_aug:
            H_in = img[0].shape[0]
            W_in = img[0].shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else: # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        '''
        # random crop
        cropper = RandomCrop(img_gt.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img_rainy = cropper(img_rainy)
        img_gt = cropper(img_gt)
        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img_rainy = np.rot90(img_rainy, rotate)
                img_gt = np.rot90(img_gt, rotate)
            if np.random.random() >= 0.5:
                img_rainy = cv2.flip(img_rainy, flipCode = 0)
                img_gt = cv2.flip(img_gt, flipCode = 0)
                
        '''        
        # add noise
        img = img.astype(np.float32) # RGB image in range [0, 255]
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise
        '''

        # normalization
        img_rainy = img_rainy.astype(np.float32) # RGB image in range [0, 255]
        img_gt = img_gt.astype(np.float32) # RGB image in range [0, 255]
        img_rainy = img_rainy / 255.0
        img_rainy = torch.from_numpy(img_rainy.transpose(2, 0, 1)).contiguous()
        img_gt = img_gt / 255.0
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).contiguous()

        return img_rainy, img_gt
    
    def __len__(self):
        return len(self.imglist)
    