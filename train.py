"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from face.Face import FaceVis, FaceModel
import os
import torch
import torch.nn as nn
import numpy as np

def recover_ori(exp):
    # exp: H X W X C

    # Normalize data
    # x mean = -0.03465565666556358
    # x std = 4.96116828918457
    # y mean = -1.4577744007110596
    # y std = 6.710243225097656
    # z mean = 5.440230369567871
    # z std = 3.930370330810547
    x_mean = torch.from_numpy(np.array([-0.0347]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    x_std = torch.from_numpy(np.array([4.9612]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    y_mean = torch.from_numpy(np.array([-1.4578]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    y_std = torch.from_numpy(np.array([6.7102]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    z_mean = torch.from_numpy(np.array([5.4402]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    z_std = torch.from_numpy(np.array([3.9304]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1

    # Adapt dimension

    ori_exp_x = exp[:, :, 0].unsqueeze(-1)*x_std+x_mean # H X W X 1
    ori_exp_y = exp[:, :, 1].unsqueeze(-1)*y_std+y_mean 
    ori_exp_z = exp[:, :, 2].unsqueeze(-1)*z_std+z_mean

    ori_exp = torch.cat((ori_exp_x, ori_exp_y, ori_exp_z), dim=-1) # H X W X 3

    return ori_exp # H X W X 3

def vis_geometry(pc_tensor, out_obj_path):
    # load template
    face_model = FaceModel()

    # convert pc to facial mesh
    face_model.update_pc(pc_tensor)

    # save to output
    face_model.export_to_obj(out_obj_path)

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                if not os.path.exists(opt.objpath):
                    os.makedirs(opt.objpath)
                for name, image in model.get_current_visuals().items():
                    save_obj_path = os.path.join(opt.objpath, "%d_%s.obj"%(total_iters, name))
                    pc_tensor = recover_ori(image[0].transpose(0, 2).transpose(0, 1).detach().cpu()).transpose(0, 2).transpose(1, 2)
                    vis_geometry(pc_tensor, save_obj_path)
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
