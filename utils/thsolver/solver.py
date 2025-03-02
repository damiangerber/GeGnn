# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------
from utils.thsolver import default_settings 
    
import os
import torch
import torch.nn
import torch.optim
import torch.utils.data
import time
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from .sampler import InfSampler
from .tracker import AverageTracker
from .config import parse_args
from .lr_scheduler import get_lr_scheduler

class Solver:
  def __init__(self, FLAGS, is_master=True):
      self.FLAGS = FLAGS
      self.is_master = is_master
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.disable_tqdm = not (is_master and FLAGS.SOLVER.progress_bar)
      self.start_epoch = 1
      
      self.model = None           
      self.optimizer = None       
      self.scheduler = None       
      self.summary_writer = None  
      self.log_file = None        
      self.eval_rst = dict()      

  def get_model(self, flags):
      raise NotImplementedError

  def get_dataset(self, flags):
      raise NotImplementedError

  def train_step(self, batch):
      raise NotImplementedError

  def test_step(self, batch):
      raise NotImplementedError

  def eval_step(self, batch):
      raise NotImplementedError
    
  def embd_decoder_func(self, i, j, embd):
      raise NotImplementedError
    
  def result_callback(self, avg_tracker: AverageTracker, epoch):
    pass  # additional operations based on the avg_tracker

  def config_dataloader(self, disable_train_data=False):
      flags_train, flags_test = self.FLAGS.DATA.train, self.FLAGS.DATA.test

      if not disable_train_data and not flags_train.disable:
          self.train_loader = self.get_dataloader(flags_train)
          self.train_iter = iter(self.train_loader)

      if not flags_test.disable:
          self.test_loader = self.get_dataloader(flags_test)
          self.test_iter = iter(self.test_loader)

  def get_dataloader(self, flags):
      dataset, collate_fn = self.get_dataset(flags)
      sampler = InfSampler(dataset, shuffle=flags.shuffle)
      data_loader = DataLoader(
          dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
          sampler=sampler, collate_fn=collate_fn, pin_memory=False)
      return data_loader

  def config_model(self):
      flags = self.FLAGS.MODEL
      model = self.get_model(flags)
      model.to(self.device)
      if self.is_master:
          print(model)
      self.model = model

  def config_optimizer(self):
      flags = self.FLAGS.SOLVER
      parameters = self.model.parameters()

      if flags.type.lower() == 'sgd':
          self.optimizer = torch.optim.SGD(
              parameters, lr=flags.lr, weight_decay=flags.weight_decay, momentum=0.9)
      elif flags.type.lower() == 'adam':
          self.optimizer = torch.optim.Adam(
              parameters, lr=flags.lr, weight_decay=flags.weight_decay)
      elif flags.type.lower() == 'adamw':
          self.optimizer = torch.optim.AdamW(
              parameters, lr=flags.lr, weight_decay=flags.weight_decay)
      else:
          raise ValueError

  def config_lr_scheduler(self):
      self.scheduler = get_lr_scheduler(self.optimizer, self.FLAGS.SOLVER)

  def configure_log(self, set_writer=True):
      self.logdir = self.FLAGS.SOLVER.logdir
      self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
      self.log_file = os.path.join(self.logdir, 'log.csv')

      if self.is_master:
          tqdm.write('Logdir: ' + self.logdir)

      if self.is_master and set_writer:
          self.summary_writer = SummaryWriter(self.logdir, flush_secs=20)
          if not os.path.exists(self.ckpt_dir):
              os.makedirs(self.ckpt_dir)

  def train_epoch(self, epoch):
    self.model.train()

    tick = time.time()
    elapsed_time = dict()
    train_tracker = AverageTracker()
    rng = range(len(self.train_loader))
    log_per_iter = self.FLAGS.SOLVER.log_per_iter
    for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
      # load data
      batch = self.train_iter.__next__()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

      elapsed_time['time/data'] = torch.Tensor([time.time() - tick])

      # forward and backward
      self.optimizer.zero_grad()
      output = self.train_step(batch)
      output['train/loss'].backward()

      # grad clip
      clip_grad = self.FLAGS.SOLVER.clip_grad
      if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

      # apply the gradient
      self.optimizer.step()

      # track the averaged tensors
      elapsed_time['time/batch'] = torch.Tensor([time.time() - tick])
      tick = time.time()
      output.update(elapsed_time)
      train_tracker.update(output)

      if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

      if log_per_iter > 0 and it % log_per_iter == 0:
        train_tracker.log(epoch, msg_tag='- ', notes=f'iter: {it}', print_time=False)

      train_tracker.log(epoch, self.summary_writer)

  def test_epoch(self, epoch):
    self.model.eval()
    test_tracker = AverageTracker()
    test_err_distribution = []
    rng = range(len(self.test_loader))
    for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
      # forward
      batch = self.test_iter.__next__()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      output = self.test_step(batch)
      try:
        test_err_distribution.append(batch['filename'][0] + " " + str(float(output['test/loss'])))
      except:
        pass
      # track the averaged tensors
      test_tracker.update(output)
      
    test_tracker.log(epoch, self.summary_writer, self.log_file, msg_tag='=>')
    self.result_callback(test_tracker, epoch)
    with open(self.logdir + "/err_statistic.txt", "w") as f:
      f.write(str(test_err_distribution))

  def eval_epoch(self, epoch):
    self.model.eval()
    eval_step = min(self.FLAGS.SOLVER.eval_step, len(self.test_loader))
    if eval_step < 1:
      eval_step = len(self.test_loader)
    for it in tqdm(range(eval_step), ncols=80, leave=False):
      batch = self.test_iter.__next__()
      batch['iter_num'] = it
      batch['epoch'] = epoch
      with torch.no_grad():
        self.eval_step(batch)


  def save_checkpoint(self, epoch):
    # save checkpoint
    model_dict = self.model.state_dict()
    ckpt_name = os.path.join(self.ckpt_dir, '%05d' % epoch)
    torch.save(model_dict, ckpt_name + '.model.pth')
    torch.save({'model_dict': model_dict, 'epoch': epoch,
                'optimizer_dict': self.optimizer.state_dict(),
                'scheduler_dict': self.scheduler.state_dict(), },
               ckpt_name + '.solver.tar')

  def load_checkpoint(self):
    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      return
    map_location = self.device  # Ensure checkpoint loads on correct device
    trained_dict = torch.load(ckpt, map_location=map_location)
    model_dict = trained_dict.get('model_dict', trained_dict)
    self.model.load_state_dict(model_dict)
    if 'epoch' in trained_dict:
      self.start_epoch = trained_dict['epoch'] + 1
    if 'optimizer_dict' in trained_dict and self.optimizer:
      self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
    if 'scheduler_dict' in trained_dict and self.scheduler:
      self.scheduler.load_state_dict(trained_dict['scheduler_dict'])

  def manual_seed(self):
    rand_seed = self.FLAGS.SOLVER.rand_seed
    if rand_seed > 0:
      random.seed(rand_seed)
      np.random.seed(rand_seed)
      torch.manual_seed(rand_seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(rand_seed)
        torch.cuda.manual_seed_all(rand_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

  def train(self):
    self.manual_seed()
    self.config_model()
    self.config_dataloader()
    self.config_optimizer()
    self.config_lr_scheduler()
    self.configure_log()
    self.load_checkpoint()

    rng = range(self.start_epoch, self.FLAGS.SOLVER.max_epoch+1)
    for epoch in tqdm(rng, ncols=80, disable=self.disable_tqdm):
      # training epoch
      self.train_epoch(epoch)

      # update learning rate
      self.scheduler.step()
      lr = self.scheduler.get_last_lr()
      self.summary_writer.add_scalar('train/lr', lr[0], epoch)

      # testing epoch
      if epoch % self.FLAGS.SOLVER.test_every_epoch == 0:
        self.test_epoch(epoch)

      # checkpoint
      self.save_checkpoint(epoch)

  def test(self):
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    self.test_epoch(epoch=0)

  def evaluate(self):
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()
    for epoch in tqdm(range(self.FLAGS.SOLVER.eval_epoch), ncols=80):
      self.eval_epoch(epoch)
      
  def profile(self):
      r''' Set `DATA.train.num_workers 0` when using this function. '''
      
      # Ensure model and dataloader are configured
      self.config_model()
      self.config_dataloader()
      
      logdir = self.FLAGS.SOLVER.logdir

      # Check PyTorch version
      version = torch.__version__.split('.')
      larger_than_110 = int(version[0]) > 0 and int(version[1]) > 10
      if not larger_than_110:
          print('The profile function is only available for Pytorch>=1.10.0.')
          return

      # Warm-up phase to ensure initial profiling works well
      batch = next(iter(self.train_loader))
      batch = {k: v.to(self.device) for k, v in batch.items()}  # Ensure data is on the right device
      for _ in range(3):
          output = self.train_step(batch)
          output['train/loss'].backward()

      # Start profiling
      with torch.profiler.profile(
              activities=[torch.profiler.ProfilerActivity.CPU,
                          torch.profiler.ProfilerActivity.CUDA],
              on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
              record_shapes=True, profile_memory=True, with_stack=True,
              with_modules=True) as prof:
          
          for i in range(3):
              output = self.train_step(batch)
              output['train/loss'].backward()
              prof.step()

      # Print the profiling results, sorted by GPU time and memory usage
      print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=10)
                .table(sort_by="cuda_time_total", row_limit=10))
      print(prof.key_averages(group_by_input_shape=True, group_by_stack_n=10)
                .table(sort_by="cuda_memory_usage", row_limit=10))


  def run(self):
    eval('self.%s()' % self.FLAGS.SOLVER.run)

  @classmethod
  def update_configs(cls):
    pass

  @classmethod
  def worker(cls, FLAGS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = cls(FLAGS, is_master=True)
    solver.device = device
    solver.run()

  @classmethod
  def main(cls):
    cls.update_configs()
    FLAGS = parse_args()
    cls.worker(FLAGS)
    
    # 做可视化
    visualizer = cls(FLAGS, is_master=True)
    ret = visualizer.get_visualization_data()
    return ret


  def get_visualization_data(self):
    """ helper function of visualization. 
        return a dict, containing the following components:
          'filename': the name of the obj file to load
          'dist_func': a function: (i, j, embd) => int
          'embedding': embedding of vertices 
        the dict will be used in interactive.py.
    Returns:
        a dict
    """
    # helper function, a utility for the visualization
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    self.load_checkpoint()

    self.model.eval()

    for it in tqdm(range(1), ncols=80, leave=False):
      batch = self.test_iter.__next__()
      with torch.no_grad():
        embedding = self.get_embd(batch)

    if default_settings.get_global_value("get_test_stat"):
      self.test_epoch(499)


    mesh_file = default_settings.get_global_value("test_mesh")
    if mesh_file == None:
      filename = batch['filename'][0]
    else:
      filename = mesh_file
    # return: filename of obj file, f function, embd of the vertices
    # specially designed for geodesic dist task. may not operate correctly on other tasks
    # dist func:
    return {
      'filename': filename,
      'dist_func': self.embd_decoder_func,
      'embedding': embedding
    }
