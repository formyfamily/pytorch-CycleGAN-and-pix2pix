from geo.Mesh import Mesh
import os
import torch
import torch.nn.functional as F
import imageio
import numpy as np


class FaceModel:
    '''
    FaceModel is the class bridging between torch-tensor uv representation of face,
    with traditional geometry representation for rendering pipelines.
    FaceModel stores all its elements using numpy array,
    and provides functions converting from tensors.
    '''

    def __init__(self, template_file=os.path.join(os.path.dirname(__file__), 'template', 'template.obj')):
        self.mesh = Mesh(path=template_file)

        # remove backside of head
        frontal_vt_flag = self.mesh.tex_coords[:, 0] < 1
        frontal_face_flag = frontal_vt_flag[self.mesh.tex_faces].sum(axis=1) == 3
        self.mesh.tex_faces = self.mesh.tex_faces[frontal_face_flag]
        self.mesh.faces = self.mesh.faces[frontal_face_flag]

        # now deal with albedo
        template_uv_file = os.path.join(os.path.dirname(__file__), 'template', 'uv.png')
        self.albedo = imageio.imread(template_uv_file)

    def update_pc(self, pc):
        '''
        Update the mesh geometry by providing pc tensor
        :param pc: [3, H, W] detached tensor of pc
        '''
        uv = self.mesh.tex_coords * 2 - 1
        # This is important since the image array has (0, 0) index on top-left corner of the image
        uv[:, 1] *= -1
        uv = torch.from_numpy(uv).float()
        uv = uv.unsqueeze(0).unsqueeze(0)
        tmp = F.grid_sample(pc.unsqueeze(0), uv)
        uv_pos = tmp[0, :, 0, :].transpose(0, 1).numpy()
        self.mesh.vertices = uv_pos
        self.mesh.faces = self.mesh.tex_faces
        self.mesh.re_calculate_normal()

    def update_al(self, al):
        '''
        Update the albedo by providing al tensor
        :param al: [3, H, W] detached tensor of al, re-normalized to [0, 1]
        '''

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy().transpose(1, 2, 0)

        self.albedo = to_numpy(al)

    def export_to_obj(self, path):
        mtl_path = path.replace('.obj', '.mtl')
        al_path = path.replace('.obj', '.albedo.png')

        # write obj
        file = open(path, 'w')
        file.write('mtllib ./%s\n' % os.path.basename(mtl_path))
        for v in self.mesh.vertices:
            file.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        for vt in self.mesh.tex_coords:
            file.write('vt %.6f %.6f\n' % (vt[0], vt[1]))

        for f, ft in zip(self.mesh.faces, self.mesh.tex_faces):
            f_plus = f + 1
            ft_plus = ft + 1
            file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], ft_plus[0], f_plus[1], ft_plus[1], f_plus[2], ft_plus[2]))
        file.close()

        # write mtl
        file = open(mtl_path, 'w')
        file.write('''newmtl Textured
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
d 1.0
illum 2
''')
        file.write('map_Ka %s\n' % os.path.basename(al_path))
        file.close()

        imageio.imwrite(al_path, self.albedo)


class FaceVis:
    gl_content = False
    resolution = 0

    def __init__(self, resolution=512):
        if not self.gl_content:
            from render.gl import create_opengl_context
            create_opengl_context(resolution, resolution)
            self.gl_content = True
            self.resolution = 512
        from render.gl import ShRender
        self.render = ShRender(self.resolution, self.resolution)

    def render_model(self, face_model):
        # Init a camera
        from render.BaseCamera import BaseCamera
        from render.CameraPose import CameraPose

        intrinsic_cam = BaseCamera()
        intrinsic_cam.set_parameters(40)
        intrinsic_cam.near = 20
        intrinsic_cam.far = 60
        extrinsic_cam = CameraPose()
        extrinsic_cam.center = [0, 0, 40]
        extrinsic_cam.sanity_check()

        vertices = face_model.mesh.vertices
        faces = face_model.mesh.faces
        normals = face_model.mesh.normals
        norm_faces = face_model.mesh.norm_faces
        tex_coords = face_model.mesh.tex_coords
        tex_faces = face_model.mesh.tex_faces

        # Here we pack the vertex data needed for the render
        vert_data = vertices[faces.reshape([-1])]
        self.render.set_attrib(0, vert_data)
        norm_data = normals[norm_faces.reshape([-1])]
        self.render.set_attrib(1, norm_data)
        uv_data = tex_coords[tex_faces.reshape([-1])]
        self.render.set_attrib(2, uv_data)

        # Here we set the textures
        self.render.set_texture('TargetTexture', face_model.albedo)

        # Use a preset sh
        sh = np.array([[0.865754, 0.880196, 0.947154],
                       [-0.205633, -0.211215, -0.250504],
                       [0.349584, 0.365084, 0.463253],
                       [-0.0789622, -0.0750734, -0.091405],
                       [0.077691, 0.0810771, 0.0922284],
                       [-0.402685, -0.40834, -0.462763],
                       [0.328907, 0.328656, 0.370725],
                       [-0.131815, -0.140992, -0.158298],
                       [-0.0992293, -0.0983686, -0.107975]])

        # Here we pack all uniform variables of the shader into a single dictionary
        model_view = extrinsic_cam.get_model_view_mat()
        persp = intrinsic_cam.get_projection_mat()
        uniform_dict = {
            'ModelMat': model_view,
            'PerspMat': persp,
            'SHCoeffs': sh,
        }
        self.render.draw(
            uniform_dict
        )

        # Finally we save the rendered results.
        # Compare with the standard output under data/tests/
        color = self.render.get_color()
        return color
