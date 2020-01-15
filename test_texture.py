"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_bs_images
from util import html
import torch
import torch.nn as nn
import numpy as np
from face.Face import FaceVis, FaceModel

def recover_ori(exp):
    # exp: H X W X C
    exp = (exp+1)/2 # 0~1

    x_min = -10.7578125
    x_max = 11.5546875
    y_min = -22.625
    y_max = 14.578125
    z_min = -6.078125
    z_max = 14.125   

    ori_exp_x = exp[:, :, 0].unsqueeze(-1)*(x_max-x_min)+x_min # H X W X 1
    ori_exp_y = exp[:, :, 1].unsqueeze(-1)*(y_max-y_min)+y_min 
    ori_exp_z = exp[:, :, 2].unsqueeze(-1)*(z_max-z_min)+z_min     

    ori_exp = torch.cat((ori_exp_x, ori_exp_y, ori_exp_z), dim=-1) # H X W X 3
    mask = torch.from_numpy(np.load("mask_128.npy"))[:, :, None] # H X W X 1

    ori_exp = ori_exp * mask

    return ori_exp # H X W X 3

def vis_geometry(pc_tensor, out_obj_path):
    # load template
    face_model = FaceModel()

    # convert pc to facial mesh
    face_model.update_pc(pc_tensor)

    # save to output
    face_model.export_to_obj(out_obj_path)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    

    opt.dataset_mode = "facetextest"
    paired_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
   
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
   
    total_iters = 0
    img_dir = "vis_cond_geo2tex_blendshapes"
    # for i, data in enumerate(dataset):
    for i, paired_data in enumerate(paired_dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        model.set_input(paired_data)  # unpack data from data loader
        
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
    
        total_iters += 1

        save_bs_images(visuals, img_dir, i, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # webpage.save()  # save the HTML