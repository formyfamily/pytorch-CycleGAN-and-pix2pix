import numpy as np
import torch


class Mesh:
    '''
    A Tri-Mesh class
    When loading from a quad mesh, it automatically converts quad mesh to tri-representation
    '''

    def __init__(self, path=None):
        self.vertices = None
        self.tex_coords = None
        self.faces = None
        self.tex_faces = None
        self.normals = None
        self.norm_faces = None

        if path is not None:
            self.read_from_obj(path)

    def re_calculate_normal(self):
        self.normals = self.compute_normal(self.vertices, self.faces)
        self.norm_faces = self.faces

    def write_to_obj(self, path):
        file = open(path, 'w')
        for v in self.vertices:
            file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        for f in self.faces:
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
        file.close()

    @staticmethod
    def compute_normal(vertices, faces):
        def normalize_v3(arr):
            ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
            lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
            eps = 0.00000001
            lens[lens < eps] = eps
            arr[:, 0] /= lens
            arr[:, 1] /= lens
            arr[:, 2] /= lens
            return arr

        # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros(vertices.shape, dtype=vertices.dtype)
        # Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[faces]
        # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        # we need to normalize these, so that our next step weights each normal equally.
        normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[faces[:, 0]] += n
        norm[faces[:, 1]] += n
        norm[faces[:, 2]] += n
        normalize_v3(norm)

        return norm

    def read_from_obj(self, path):
        v_data = []
        vt_data = []

        face_v_data = []
        face_vt_data = []

        f = open(path, "r")
        for line in f:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                v_data.append(v)
            elif values[0] == 'vt':
                vt = list(map(float, values[1:3]))
                vt_data.append(vt)

            elif values[0] == 'f':
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    face_v_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                    face_v_data.append(f)
                # tri mesh
                else:
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    face_v_data.append(f)

                # deal with texture
                if len(values[1].split('/')) >= 2:
                    # quad mesh
                    if len(values) > 4:
                        f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                        face_vt_data.append(f)
                        f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                        face_vt_data.append(f)
                    # tri mesh
                    elif len(values[1].split('/')[1]) != 0:
                        f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                        face_vt_data.append(f)

        vertices = np.array(v_data)
        faces = np.array(face_v_data)
        faces[faces > 0] -= 1

        tex_coords = np.array(vt_data)
        tex_faces = np.array(face_vt_data)
        tex_faces[tex_faces > 0] -= 1

        self.vertices = vertices
        self.tex_coords = tex_coords
        self.faces = faces
        self.tex_faces = tex_faces
        self.re_calculate_normal()