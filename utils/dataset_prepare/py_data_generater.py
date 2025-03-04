import os
import multiprocessing as mp
import random

import numpy as np
import trimesh
import pygeodesic.geodesic as geodesic
from tqdm import tqdm

from utils.thsolver import default_settings
from utils.thsolver.config import parse_args

# Initialize global settings
default_settings._init()
FLAGS = parse_args()
default_settings.set_global_values(FLAGS)


def visualize_ssad(vertices: np.ndarray, triangles: np.ndarray, source_index: int):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = None
    distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
    return distances


def visualize_two_pts(vertices: np.ndarray, triangles: np.ndarray, source_index: int, dest_index: int):
    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    # Define the source and target point ids with respect to the points array
    source_indices = np.array([source_index])
    target_indices = np.array([dest_index])
    distances, _ = geoalg.geodesicDistance(source_indices, target_indices)
    return distances


def data_prepare_gen_dataset(object_file: str, output_path: str, num_sources, num_destinations, tqdm_on=True):
    vertices = []
    triangles = []

    with open(object_file, "r") as f:
        lines = f.readlines()
        for each in lines:
            if each.startswith("v "):
                temp = each.split()
                vertices.append([float(temp[1]), float(temp[2]), float(temp[3])])
            if each.startswith("f "):
                temp = each.split()
                # 
                temp[3] = temp[3].split("/")[0]
                temp[1] = temp[1].split("/")[0]
                temp[2] = temp[2].split("/")[0]
                triangles.append([int(temp[1]) - 1, int(temp[2]) - 1, int(temp[3]) - 1])
    vertices = np.array(vertices)
    triangles = np.array(triangles)


    # Initialise the PyGeodesicAlgorithmExact class instance
    geoalg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)

    result = np.array([[0,0,0]])

    sources = np.random.randint(low=0, high=len(vertices), size=[num_sources])

    # iterate
    # only the process on process #0 will be displayed
    # this should not be problematic or confusing on most homogeneous CPUs
    it = tqdm(range(num_sources)) if tqdm_on else range(num_sources)
    for i in it:
        source_indices = np.array([sources[i]])
        target_indices = np.random.randint(low=0, high=len(vertices), size=[num_destinations])
        if (source_indices.max() >= len(vertices)) or (target_indices.max() >= len(vertices)):
            raise ValueError("Index out of range")

        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)

        a = source_indices.repeat([num_destinations]).reshape([-1,1])
        b = target_indices.reshape([-1,1])
        c = distances.reshape([-1,1])
        new = np.concatenate([a, b, c], -1)
        result = np.concatenate([result, new])

    np.savetxt(output_path, result)

def computation_thread(filename, object_name, train_src, train_tgt, test_src, test_tgt, idx=None):
    assert idx is not None, "An index ('idx') has to be given"
    tqdm_on = False
    if idx == 0:
        tqdm_on = True
    print(filename, object_name)
    
    train_output = object_name + "_train_" + str(idx)
    test_output = object_name + "_test_" + str(idx)
    
    data_prepare_gen_dataset(filename, train_output, train_src, train_tgt, tqdm_on=tqdm_on)
    data_prepare_gen_dataset(filename, test_output, test_src, test_tgt, tqdm_on=False)

# def computation_thread(filename, object_name, a, b, c, d, idx=None):
#     assert idx != None, "an idx has to be given"
#     tqdm_on = False
#     if idx == 0:
#         tqdm_on = True
#     print(filename, object_name)
#     data_prepare_gen_dataset(filename, object_name + "_train_" + str(idx), a, b, tqdm_on=tqdm_on)


if __name__ == "__main__":
    """
    Generates the npz files and filelists for the mesh data. Samples the geodesic distances for both training and test datasets.
    
    Args:
        PATH_TO_MESH: str, path to the mesh files
        PATH_TO_OUTPUT_NPZ: str, path to the output npz files
        PATH_TO_OUTPUT_FILELIST: str, path to the output filelist
        
        split_ratio: float, ratio of training data
        
        num_train_sources: int, training set: number of sources to sample
        num_train_targets_per_source: int, training set: destinations per source
        num_test_sources: int, testing set: number of sources to sample
        num_test_targets_per_source: int, testing set: destinations per source
        
        file_size_threshold: int, file size threshold
        threads: int, number of threads. 0 uses all cores
    """

    PATH_TO_MESH = FLAGS.DATA.preparation.path_to_mesh
    PATH_TO_OUTPUT_NPZ = FLAGS.DATA.preparation.path_to_output_npz
    PATH_TO_OUTPUT_FILELIST = FLAGS.DATA.preparation.path_to_output_filelist

    SPLIT_RATIO = FLAGS.DATA.preparation.split_ratio
    FILE_SIZE_THRESHOLD = FLAGS.DATA.preparation.file_size_threshold
    LARGE_DISTANCE_THRESHOLD = 1e8
    
    # number of sources and targets for sampling the distances
    num_train_sources = FLAGS.DATA.preparation.num_train_sources
    num_train_targets_per_source = FLAGS.DATA.preparation.num_train_targets_per_source
    num_test_sources = FLAGS.DATA.preparation.num_test_sources
    num_test_targets_per_source = FLAGS.DATA.preparation.num_test_targets_per_source
    
    threads = FLAGS.DATA.preparation.threads
    object_name = None
    
    assert threads >= 0 and type(threads) == int
    
    if threads == 0:
        threads = mp.cpu_count()
        print(f"Automatically utilize all CPU cores ({threads})")
    else:
        print(f"{threads} CPU cores are utilized!")

    # make dirs, if not exist
    if not os.path.exists(PATH_TO_OUTPUT_NPZ):
        os.mkdir(PATH_TO_OUTPUT_NPZ)
    if not os.path.exists(PATH_TO_OUTPUT_FILELIST):
        os.mkdir(PATH_TO_OUTPUT_FILELIST)
            
    all_files = []
    for mesh in os.listdir(PATH_TO_MESH):
        # check if the file is too large
        if os.path.getsize(PATH_TO_MESH + mesh) < FILE_SIZE_THRESHOLD:
            all_files.append(os.path.join(PATH_TO_MESH, mesh))

    object_names = all_files
    for i in range(len(object_names)):
        if object_names[i].endswith(".obj"):
            object_names[i] = object_names[i][:-4]
            
    print(f"Current dir: {os.getcwd()}, object to be processed: {len(object_names)}")
   
    # handle the case when the output file already exists
    for i in tqdm(range(len(object_names))):
        object_name = object_names[i]
        if object_name.split("/")[-1][0] == ".":
            continue        # not an obj file
        
        filename_out = PATH_TO_OUTPUT_NPZ + object_name + ".npz"
        if os.path.exists(filename_out):
            continue
        
        filename = object_name + ".obj"
       
        train_data_filename_list = []
        test_data_filename_list = []

        pool = []

        for t in range(threads):
            task = mp.Process(target=computation_thread, args=(filename, object_name, num_train_sources//threads,num_train_targets_per_source,num_test_sources//threads,num_train_targets_per_source, t,))
            task.start()
            pool.append(task)
        for t, task in enumerate(pool):
            task.join()
            train_data_filename_list.append(object_name + "_train_" + str(t))

        try:
            for i in range(len(train_data_filename_list)):
                # train data
                with open(object_name + "_train_" + str(i), "r") as f:
                    data = f.read()
                with open(object_name + "_train", "a") as f:
                    f.write(data)
        except:
            raise ValueError("Error on " + object_name + ", this is mostly due to non-manifold (failed to initialise the PyGeodesicAlgorithmExact class instance)")

        for each in (train_data_filename_list + test_data_filename_list):
            os.remove(each)

        filename_in = object_name + ".obj"
        dist_in = object_name + "_train"
        filename_out = PATH_TO_OUTPUT_NPZ + object_name.split("/")[-1] + ".npz"
        try:
            mesh = trimesh.load_mesh(filename_in)
            dist = np.loadtxt(dist_in)
        except Exception:
            print(f"load {filename_in} or {dist_in} failed...")
            continue
        
        # delete the dist_in
        os.remove(dist_in)

        aa = mesh.edges_unique
        bb = np.concatenate([aa[:, 1:], aa[:, :1]], 1)
        cc = np.concatenate([aa, bb])
        
        # sanity check
        vertices = mesh.vertices
        if dist.max() > LARGE_DISTANCE_THRESHOLD:
            raise ValueError("Distance too large!")
        elif ((dist.astype(np.float32).max()) >= vertices.shape[0]):
            raise ValueError("Encountered a trimesh loading error!")

        np.savez(filename_out,
                edges=cc,
                vertices=mesh.vertices.astype(np.float32),
                normals=mesh.vertex_normals.astype(np.float32),
                faces=mesh.faces.astype(np.float32),
                dist_val=dist[:, 2:].astype(np.float32),
                dist_idx=dist[:, :2].astype(np.uint16),
        )
        
    
    print("\nnpz data generation finished. Now generating filelist...\n")
    lines = [] # filelist
    for each in tqdm(object_names):
        filename_out = PATH_TO_OUTPUT_NPZ + each.split("/")[-1] + ".npz"
        try:
            dist = np.load(filename_out)
            
            # sanity check
            if dist['dist_val'].max() != np.inf and dist['dist_val'].max() < LARGE_DISTANCE_THRESHOLD:
                lines.append(filename_out + "\n")
            else:
                raise ValueError("File contains inf for the distances!")
        except Exception:
            raise ValueError("load " + filename_out + " failed...")
    
    # Split up into test and train set
    random.shuffle(lines)
    train_num = int(len(lines) * SPLIT_RATIO)
    test_num  = len(lines) - train_num
    train_lines = lines[:train_num]
    test_lines  = lines[train_num:]
    
    with open(PATH_TO_OUTPUT_FILELIST + 'filelist_train.txt', 'w') as f:
        f.writelines(train_lines)
        
    with open(PATH_TO_OUTPUT_FILELIST + 'filelist_test.txt', 'w') as f:
        f.writelines(test_lines)