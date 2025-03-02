import numpy as np
import torch
import torch.nn as nn
import hgraph
import trimesh
import argparse

# a function that reads a triangular mesh, and generates its corresponding graph
def read_mesh(path, to_tensor=True, device="cpu"):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    edges = mesh.edges_unique
    edges_reversed = np.concatenate([edges[:, 1:], edges[:, :1]], 1)
    edges = np.concatenate([edges, edges_reversed], 0)
    edges = np.transpose(edges)
    normals = mesh.vertex_normals
    norm_normals = np.linalg.norm(normals, axis=1)
    normals = normals / norm_normals[:, np.newaxis]
    faces = mesh.faces
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    
    if to_tensor:
        vertices = torch.from_numpy(vertices).float().to(device)
        edges = torch.from_numpy(edges).long().to(device)
        normals = torch.from_numpy(np.array(normals)).float().to(device)
        faces = torch.from_numpy(faces).long().to(device)
        face_normals = torch.from_numpy(np.array(face_normals)).float().to(device)
        face_areas = torch.from_numpy(np.array(face_areas)).float().to(device)
    
    return {
        "vertices": vertices,
        "edges": edges,
        "normals": normals,
        "faces": faces,
        "face_normals": face_normals,
        "face_areas": face_areas,
    }

# a wrapper of pretrained model
class PretrainedModel(nn.Module):
    def __init__(self, ckpt_path):
        super(PretrainedModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = hgraph.models.graph_unet.GraphUNet(6, 256, None, None).to(self.device)
        self.embds = None
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_dict'])
        
    def embd_decoder_func(self, i, j, embedding):
        i = i.long()
        j = j.long()
        embd_i = embedding[i].squeeze(-1)
        embd_j = embedding[j].squeeze(-1)
        embd = (embd_i - embd_j) ** 2
        pred = self.model.embedding_decoder_mlp(embd)
        return pred.squeeze(-1)
    
    def precompute(self, mesh):
        with torch.no_grad():
            vertices = mesh['vertices']
            normals = mesh['normals']
            edges = mesh['edges']
            tree = hgraph.hgraph.HGraph()
            tree.build_single_hgraph(
                hgraph.hgraph.Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges)
            )
            self.embds = self.model(
                torch.cat([vertices, normals], dim=1), 
                tree, 
                tree.depth, 
                dist=None,
                only_embd=True
            ).detach()
    
    def forward(self, p_vertices=None, q_vertices=None):
        assert self.embds is not None, "Please call precompute() first!"
        with torch.no_grad():
            return self.embd_decoder_func(p_vertices, q_vertices, self.embds)
    
    def SSAD(self, source: list):
        assert self.embds is not None, "Please call precompute() first!"
        ret = []
        with torch.no_grad():
            for src in source:
                s = torch.tensor([src]).repeat(self.embds.shape[0]).to(self.device)
                t = torch.arange(self.embds.shape[0]).to(self.device)
                ret.append(self.embd_decoder_func(s, t, self.embds))
        return ret


# a wrapper of pretrained model, so that it can be called directly from the command line
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='SSAD', 
                        help='only SSAD available for now')
    parser.add_argument('--test_file', type=str, default=None, help='path to the obj file')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to the checkpoint')
    parser.add_argument('--start_pts', type=str, default=None, help='an int is expected.')
    parser.add_argument('--output', type=str, default=None, help='path to the output file')
    args = parser.parse_args()
    
    if args.mode == "SSAD":
        obj_dic = read_mesh(args.test_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print the vertex and face number
        print("Vertex number: ", obj_dic['vertices'].shape[0], "Face number: ", obj_dic['faces'].shape[0])
        start_pts = torch.tensor(int(args.start_pts)).to(device)
        
        model = PretrainedModel(args.ckpt_path).to(device)
        model.precompute(obj_dic)
        dist_pred = model.SSAD([start_pts])[0]
        np.save(args.output, dist_pred.detach().cpu().numpy())
        # save the colored mesh for visualization
        # given the vertices, faces of a mesh, save it as obj file
        def save_mesh_as_obj(vertices, faces, scalar=None, path="out/our_mesh.obj"):
            with open(path, 'w') as f:
                f.write('# mesh\n')     # header of LittleRender
                for v in vertices:
                    f.write('v ' + str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + '\n')
                for face in faces:
                    f.write('f ' + str(face[0]+1) + ' ' + str(face[1]+1) + ' ' + str(face[2]+1) + '\n')
                if scalar is not None:
                    # normalize the scalar to [0, 1]
                    scalar = (scalar - np.min(scalar)) / (np.max(scalar) - np.min(scalar))
                    for c in scalar:
                        f.write('c ' + str(c) + ' ' + str(c) + ' ' + str(c) + '\n')
                        
            print("Saved mesh as obj file:", path, end="")
            if scalar is not None:
                print(" (with color) ")
            else:
                print(" (without color)")
                
        save_mesh_as_obj(obj_dic['vertices'].detach().to(device).numpy(), 
                         obj_dic['faces'].detach().to(device).numpy(), 
                         dist_pred.detach().to(device).numpy())
        

    else:
        print("Invalid mode! (" + args.mode + ")")
    
    
        
if __name__ == "__main__":
    main()
    
    ###################################
    # visualization via polyscope starts
    # comment out the following lines if you are using ssh
    ###################################
    import polyscope as ps
    import numpy as np
    import trimesh

    # load mesh
    mesh = trimesh.load_mesh("out/our_mesh.obj", process=False)
    vertices = mesh.vertices
    faces = mesh.faces

    # load numpy array
    colors = np.load("out/ssad_ours.npy")
    print(colors.shape)

    # Initialize polyscope
    ps.init()
    ps_cloud = ps.register_point_cloud("my mesh", vertices)
    ps_cloud.add_scalar_quantity("geo_distance", colors, enabled=True)
    ps.show()
    ###################################
    # visualization via polyscope ends
    ###################################
