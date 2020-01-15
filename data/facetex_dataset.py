import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import imageio
import random
import torch
import numpy as np
import json

def load_img(filename):
    _format = 'EXR-FI' if filename.split(".")[-1] == "exr" else None
    return imageio.imread(filename, format=_format)

def scale_to_range(exp):
    x_min = -10.7578125
    x_max = 11.5546875
    y_min = -22.625
    y_max = 14.578125
    z_min = -6.078125
    z_max = 14.125

    # Normalize to 0~1
    normalized_exp_x = (exp[:, :, 0].unsqueeze(-1)-x_min)/(x_max-x_min) # H X W X 1
    normalized_exp_y = (exp[:, :, 1].unsqueeze(-1)-y_min)/(y_max-y_min)
    normalized_exp_z = (exp[:, :, 2].unsqueeze(-1)-z_min)/(z_max-z_min)

    normalized_exp = torch.cat((normalized_exp_x, normalized_exp_y, normalized_exp_z), dim=-1) # H X W X 3

    scaled_exp = normalized_exp*2-1 # Scale to -1 ~1

    return scaled_exp

class FacetexDataset(BaseDataset):
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

        self.ids_dic = json.load(open(opt.jsonfile, 'r'))

        self.ids = []
        for k in self.ids_dic.keys():
            self.ids.append(int(k))

        # self.ls_geo_folder = "/home/ICT2000/jli/local/data/LightStageFaceDB/256/PointCloud_Aligned"
        # self.tr_geo_folder = "/home/ICT2000/jli/local/data/InfiniteRealities_Triplegangers/256/PointCloud_Aligned"

        # self.ls_tex_folder = "/home/ICT2000/jli/local/data/LightStageFaceDB/256/DiffuseAlbedo"
        # self.tr_tex_folder = "/home/ICT2000/jli/local/data/InfiniteRealities_Triplegangers/256/DiffuseAlbedo"
        self.ls_geo_folder = "/home/ICT2000/jli/local/data/LightStageFaceDB/128/PointCloud_Aligned"
        self.tr_geo_folder = "/home/ICT2000/jli/local/data/InfiniteRealities_Triplegangers/128/PointCloud_Aligned"

        self.ls_tex_folder = "/home/ICT2000/jli/local/data/LightStageFaceDB/128/DiffuseAlbedo"
        self.tr_tex_folder = "/home/ICT2000/jli/local/data/InfiniteRealities_Triplegangers/128/DiffuseAlbedo"

        self.mask = torch.FloatTensor(np.load("mask_128.npy"))[:,:,None]

    def load_single_subject(self, index, exp_idx, load_geo=True):
        source = self.ids_dic[index]['source']
        f_tag_list = self.ids_dic[index]['f_tag'] # Already sorted

        if load_geo:
            postfix_str = "_pointcloud.exr"
        else:
            postfix_str = "_diffuse_albedo.exr"

        if source == "ls":
            if load_geo:
                expression_folder = self.ls_geo_folder
            else:
                expression_folder = self.ls_tex_folder
        elif source == "tr":
            if load_geo:
                expression_folder = self.tr_geo_folder
            else:
                expression_folder = self.tr_tex_folder

        f_tag = f_tag_list[exp_idx]
        expression_f_name = f_tag + postfix_str
        expression_f_path = os.path.join(expression_folder, expression_f_name)
        exp_img = torch.from_numpy(load_img(expression_f_path)).float() # H X W X C
        
        if load_geo:
            scaled_exp = scale_to_range(exp_img)
            masked_scaled_img = self.mask*scaled_exp
        else:
            masked_scaled_img = exp_img

        return masked_scaled_img.transpose(0, 2).transpose(1, 2) # C X H X W

    def load_neutral_face(self, index, load_geo=True):
        if self.ids_dic[index]['source'] == "template":
            neutral_face = self.load_single_subject(index, 0, load_geo)
        elif self.ids_dic[index]['source'] == "ls":
            neutral_face = self.load_single_subject(index, 1, load_geo)
        elif self.ids_dic[index]['source'] == "tr":
            neutral_face = self.load_single_subject(index, 0, load_geo)

        return neutral_face 
        
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
        index %= len(self.ids)
        if index == 0:
            return self.__getitem__(index+1) # Skip the template since template does not have corresponding textures
        p_a_idx = str(index)

        A_neutral_tex = self.load_neutral_face(p_a_idx, load_geo=False)
        
        num_expressions = len(self.ids_dic[p_a_idx]['f_tag'])
        exp_idx = np.random.randint(num_expressions) # Index for single expression
        A_geo = self.load_single_subject(p_a_idx, exp_idx, load_geo=True) # C X H X W
        A_tex = self.load_single_subject(p_a_idx, exp_idx, load_geo=False)
           
        return {'A': torch.cat((A_neutral_tex, A_geo), dim=0), 'B': A_tex}

    def __len__(self):
        return len(self.ids)*20