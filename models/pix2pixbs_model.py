import torch
from .base_model import BaseModel
from . import networks


class Pix2PixBSModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(norm='instance', netG='unet_256')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.no_gan = opt.no_gan 
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.no_gan:
            self.loss_names = ['G_L1']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['real_A_input', 'A_neutral', 'B_neutral', 'fake_B', 'real_B', \
        'offset_A', 'offset_B', 'offset_fake_B', 'simple_B', 'composed_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define alpha params, assume each type of expression has same alpha for different subjects
        self.alpha_mat = torch.nn.Parameter(torch.rand(26+19, 55).cuda()) # 45 X 55

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc//3 + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G.add_param_group({"params": self.alpha_mat})
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input, pair_flag=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if pair_flag: # paired data training
            self.real_A = input['A'].to(self.device) # 1 X 9 X H X W
            self.real_A_input = self.real_A[:, 0:3] # 1 X 3 X H X W
            self.A_neutral = self.real_A[:, 3:6]
            self.B_neutral = self.real_A[:, 6:9]
            self.real_B = input['B'].to(self.device) # 1 X 3 X H X W
            self.simple_B = self.real_A_input-self.A_neutral+self.B_neutral
            self.l1_mask = input["l1_mask"][:,None,None,None].float().to(self.device) # 1 X 1 X 1 X 1
        else: # blendshape data training
            self.real_A = input['A'].to(self.device) # 1 X 55 X 9 X H X W
            self.real_A_input = self.real_A[:, :, 0:3, :, :] # blendshapes BS(1) X 55 X 3 X H X W
            self.A_neutral = self.real_A[:, :, 3:6, :, :] # template neutral face, 1 X 55 X 3 X H X W
            self.B_neutral = self.real_A[:, :, 6:9, :, :] # subject neutral face, 1 X 55 X 3 X H X W
            self.real_B = input['B'].to(self.device) # 1 X 3 X H X W, random selected subject expression
            self.simple_B = self.real_A_input-self.A_neutral+self.B_neutral # 1 X 55 X 3 X H X W, subject neutral + template bs offsets
            self.alpha_idx = input['alpha'] # BS(1) indicate which line in self.alpha_mat we should use

    def forward(self, pair_flag=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if pair_flag:
            self.fake_B = self.netG(self.real_A)  # G(A) 1 X 3 X H X W
            self.offset_A = (self.real_A_input - self.A_neutral)*10
            self.offset_B = (self.real_B - self.B_neutral)*10
            self.offset_fake_B = (self.fake_B - self.B_neutral)*10
        else:
            batch_size, num_blend, c, h, w = self.real_A.size()
            self.fake_B = self.netG(self.real_A.view(batch_size*num_blend, c, h, w)).view(batch_size, num_blend, c//3, h, w)  # G(A) 1 X 55 X 3 X H X W
            self.offset_A = (self.real_A_input - self.A_neutral)*10 # Template blendshape offsets 1 X 55 X 3 X H X W
            self.offset_B = (self.real_B.unsqueeze(0) - self.B_neutral)*10 # useless in current case
            self.offset_fake_B = (self.fake_B - self.B_neutral)*10 # 1 X 55 X 3 X H X W, personalized blendshape offsets

            alpha_params = self.alpha_mat[self.alpha_idx, :].squeeze(0).clamp(min=0, max=2) # 55
            composed_offsets = (self.fake_B-self.B_neutral).squeeze(0)*alpha_params[:, None, None, None] # 55 X 3 X H X W
            self.composed_B = self.B_neutral.squeeze(0)[0, :, :, :] + composed_offsets.sum(dim=0) # 3 X H X W
            self.composed_B = self.composed_B.unsqueeze(0)

    def backward_D(self, pair_flag=True):
        """Calculate GAN loss for the discriminator"""
        if pair_flag:
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.offset_A, self.offset_fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.offset_A, self.offset_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            if self.l1_mask[0,0,0,0] == 1:
                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            else:
                self.loss_D = self.loss_D_fake
            self.loss_D.backward()
        else:
            # Fake; stop backprop to the generator by detaching fake_B
            fake_AB = torch.cat((self.offset_A.squeeze(0), self.offset_fake_B.squeeze(0)), 1)  
            # we use conditional GANs; we need to feed both input and output to the discriminator
            # 55 X 6 X H X W
            pred_fake = self.netD(fake_AB.detach()) # 55 X 1 X 30 X 30
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_AB = torch.cat((self.offset_A.squeeze(0), self.offset_B.squeeze(0)), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True) # not used in current case
            # combine loss and calculate gradients
            self.loss_D = self.loss_D_fake
            self.loss_D.backward()

    def backward_G(self, pair_flag=True):
        """Calculate GAN and L1 loss for the generator"""
        if pair_flag:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.offset_A, self.offset_fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B * self.l1_mask, self.real_B * self.l1_mask) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
            self.loss_G.backward()

            # Just for visual, not used here
            self.composed_B = self.fake_B
        else:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.offset_A.squeeze(0), self.offset_fake_B.squeeze(0)), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            # alpha_params = self.alpha_mat[self.alpha_idx, :].squeeze(0).clamp(min=0, max=2) # 55
            # composed_offsets = (self.fake_B-self.B_neutral).squeeze(0)*alpha_params[:, None, None, None] # 55 X 3 X H X W
            # self.composed_B = self.B_neutral.squeeze(0)[0, :, :, :] + composed_offsets.sum(dim=0) # 3 X H X W
            # self.composed_B = self.composed_B.unsqueeze(0)
            self.loss_G_L1 = self.criterionL1(self.composed_B, self.real_B) * self.opt.lambda_L1
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
            self.loss_G.backward()
    
    def backward_G_no_gan(self, pair_flag=True):
        """Calculate GAN and L1 loss for the generator"""
        if pair_flag:
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B * self.l1_mask, self.real_B * self.l1_mask) * self.opt.lambda_L1
            # self.loss_G_GAN = self.loss_G_L1 # Just for not changing visualization setting
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1
            self.loss_G.backward()

            # Just for visual, not used here
            self.composed_B = self.fake_B
        else:
            # Second, G(A) = B
            alpha_params = self.alpha_mat[self.alpha_idx, :].squeeze(0).clamp(min=0, max=2) # 55
            composed_offsets = (self.fake_B-self.B_neutral).squeeze(0)*alpha_params[:, None, None, None] # 55 X 3 X H X W
            self.composed_B = self.B_neutral.squeeze(0)[0, :, :, :] + composed_offsets.sum(dim=0) # 3 X H X W
            self.composed_B = self.composed_B.unsqueeze(0)
            self.loss_G_L1 = self.criterionL1(self.composed_B, self.real_B) * self.opt.lambda_L1
            # self.loss_G_GAN = self.loss_G_L1 # Just for not changing visualization setting
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_L1
            self.loss_G.backward()

    def optimize_parameters(self, pair_flag=True):
        self.forward(pair_flag)                   # compute fake images: G(A)
        if self.no_gan:
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G_no_gan(pair_flag)                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
        else:
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D(pair_flag)                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(pair_flag)                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
