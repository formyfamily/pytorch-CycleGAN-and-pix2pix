import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import imageio
import random
import torch
import numpy as np

def normalize(exp):
    # exp: H X W X C

    # Normalize data
    # x mean = -0.03465565666556358
    # x std = 4.96116828918457
    # y mean = -1.4577744007110596
    # y std = 6.710243225097656
    # z mean = 5.440230369567871
    # z std = 3.930370330810547
    x_mean = torch.from_numpy(np.array([-0.03465565666556358]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    x_std = torch.from_numpy(np.array([4.96116828918457]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    y_mean = torch.from_numpy(np.array([-1.4577744007110596]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    y_std = torch.from_numpy(np.array([6.710243225097656]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    z_mean = torch.from_numpy(np.array([5.440230369567871]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1
    z_std = torch.from_numpy(np.array([3.930370330810547]*256*256).reshape((256, 256))).unsqueeze(-1).float() # H X W X 1

    # Adapt dimension

    normalized_exp_x = (exp[:, :, 0].unsqueeze(-1)-x_mean)/x_std # H X W X 1
    normalized_exp_y = (exp[:, :, 1].unsqueeze(-1)-y_mean)/y_std
    normalized_exp_z = (exp[:, :, 2].unsqueeze(-1)-z_mean)/z_std

    normalized_exp = torch.cat((normalized_exp_x, normalized_exp_y, normalized_exp_z), dim=-1) # H X W X 3

    return normalized_exp # H X W X 3

def scale_to_range(exp):
    # x_min =  -10.75, x_max = 11.5546875, y_min = -22.625, y_max = 14.578125, z_min = -6.078125, z_max = 14.125
    image_size = 256
    x_min = torch.from_numpy(np.array([-10.7578125]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1
    x_max = torch.from_numpy(np.array([11.5546875]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1
    y_min = torch.from_numpy(np.array([-22.625]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1
    y_max = torch.from_numpy(np.array([14.578125]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1
    z_min = torch.from_numpy(np.array([-6.078125]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1
    z_max = torch.from_numpy(np.array([14.125]*image_size*image_size).reshape((image_size, image_size))).unsqueeze(-1).float() # H X W X 1

    # Normalize to 0~1
    normalized_exp_x = (exp[:, :, 0].unsqueeze(-1)-x_min)/(x_max-x_min) # H X W X 1
    normalized_exp_y = (exp[:, :, 1].unsqueeze(-1)-y_min)/(y_max-y_min)
    normalized_exp_z = (exp[:, :, 2].unsqueeze(-1)-z_min)/(z_max-z_min)

    normalized_exp = torch.cat((normalized_exp_x, normalized_exp_y, normalized_exp_z), dim=-1) # H X W X 3

    scaled_exp = normalized_exp*2-1 # Scale to -1 ~1

    return scaled_exp

class FacexDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, "/home/ICT2000/jli/local/data/Blendshapes_256_exr")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, "/home/ICT2000/jli/local/data/LightStageFaceDB/256/PointCloud_Aligned")  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, "/home/ICT2000/jli/local/data/LightStageFaceDB/256/PointCloud_Aligned")  # create a path '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, "/home/ICT2000/jli/local/data/LightStageFaceDB/256/DiffuseAlbedo")

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size, prefix="20191002_RyanWatson"))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, prefix="20190429_MichaelTrejo"))    # load images from '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.mask = torch.FloatTensor(np.load("mask.npy"))
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        index_B = index % self.A_size # For pix2pix training
        B_path = self.B_paths[index_B]
        # A = normalize(torch.FloatTensor(imageio.imread(A_path))).transpose(0, 2).transpose(1, 2) * self.mask
        # B = normalize(torch.FloatTensor(imageio.imread(B_path))).transpose(0, 2).transpose(1, 2) * self.mask

        # A = scale_to_range(torch.FloatTensor(imageio.imread(A_path))).transpose(0, 2).transpose(1, 2) * self.mask
        # B = scale_to_range(torch.FloatTensor(imageio.imread(B_path))).transpose(0, 2).transpose(1, 2) * self.mask

        B = scale_to_range(torch.FloatTensor(imageio.imread(B_path))).transpose(0, 2).transpose(1, 2) * self.mask
        A = torch.FloatTensor(imageio.imread(A_path)).transpose(0, 2).transpose(1, 2)

        # import pdb
        # pdb.set_trace()
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
