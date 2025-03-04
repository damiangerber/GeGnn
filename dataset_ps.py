import torch
from utils import ocnn

import numpy as np
from utils.thsolver import Dataset

from hgraph.hgraph import Data
from hgraph.hgraph import HGraph


class Transform(ocnn.dataset.Transform):

  def __call__(self, sample: dict, idx: int):
    """ 
    Is a transformation function that is applied to each sample. 
    The transformation randomly resamples the distances used for the prediction.
    
    Args:
      sample: a dict with keys: vertices, normals, edges, dist_idx, dist_val
      idx: the index of the sample
      
    Returns:
      a dict with keys: hgraph, vertices, normals, dist, edges
    """
    vertices = torch.from_numpy(sample['vertices'].astype(np.float32))
    normals = torch.from_numpy(sample['normals'].astype(np.float32))
    edges = torch.from_numpy(sample['edges'].astype(np.float32)).t().contiguous().long()
    dist_idx = sample['dist_idx'].astype(np.float32)
    dist_val = sample['dist_val'].astype(np.float32)
    
    dist = np.concatenate([dist_idx, dist_val], -1)
    dist = torch.from_numpy(dist)

    # randomly sample distance pairs
    size = min(len(dist), 100_000)
    rnd_idx = torch.randint(low=0, high=dist.shape[0], size=(size,))
    dist = dist[rnd_idx]

    # normalize
    norm2 = torch.sqrt(torch.sum(normals ** 2, dim=1, keepdim=True))
    normals = normals / torch.clamp(norm2, min=1.0e-12)

    # construct hierarchical graph
    h_graph = HGraph()
    h_graph.build_single_hgraph(Data(x=torch.cat([vertices, normals], dim=1), edge_index=edges))

    return {'hgraph': h_graph,
            'vertices': vertices, 'normals': normals,
            'dist': dist, 'edges': edges}


def collate_batch(batch: list):
  """  
  This function is used to collate a batch of samples.
  
  Args:
    batch: list of single samples. Each sample is a dict with keys: edges, vertices, normals, dist
    
  Returns:
    outputs: a big sample as a dict with keys: edges, vertices, normals, dist, feature, hgraph
  """
  assert type(batch) == list

  outputs = {}
  for key in batch[0].keys():
    outputs[key] = [b[key] for b in batch]

  pts_num = torch.tensor([pts.shape[0] for pts in outputs['vertices']])
  cum_sum = ocnn.utils.cumsum(pts_num, dim=0, exclusive=True)
  for i, dist in enumerate(outputs['dist']):
    dist[:, :2] += cum_sum[i]

  outputs['dist'] = torch.cat(outputs['dist'], dim=0)

  # input feature 
  vertices = torch.cat(outputs['vertices'], dim=0)
  normals = torch.cat(outputs['normals'], dim=0)
  feature = torch.cat([vertices, normals], dim=1)
  outputs['feature'] = feature

  # merge a batch of hgraphs into one super hgraph
  hgraph_super = HGraph(batch_size=len(batch))
  hgraph_super.merge_hgraph(outputs['hgraph'])
  outputs['hgraph'] = hgraph_super

  return outputs


def get_dataset(flags):
  transform = Transform(**flags)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=np.load, take=flags.take)
  return dataset, collate_batch
