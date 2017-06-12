from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import tensorflow as tf

import deepmind_lab


class CuriousAgent(object):
    def __init__(self, action_spec):
        self.action_spec = action_spec
        print('Starting CuriousAgent. Action spec:', action_spec)
        
    def step(self, reward, frame):
        self.rewards += reward
        
        return self.clip_action(self.action)
    
    def clip_action(self, action):
        return np.clip(action, self.mins, self.maxs).astype(np.intc)

    def reset(self):
        self.velocity = np.zeros([len(self.action_spec)])
        self.action = np.zeros([len(self.action_spec)])

def run(length, width, height, fps, level):
  """Spins up an environment and runs the tensorflow agent."""
  env = deepmind_lab.Lab(
      level, ['RGB_INTERLACED'],
      config={
          'fps': str(fps),
          'width': str(width),
          'height': str(height)
      })

  env.reset()

  agent = CuriousAgent(env.action_spec())
  
  reward = 0
  for _ in xrange(length):
    if not env.is_running():
      print('Environment stopped early')
      env.reset()
      agent.reset()
    obs = env.observations()
    pixels = obs['RGB_INTERLACED']
    action = agent.step(reward, pixels)
    print(action)
    reward = env.step(action, num_steps=1)

  print('Finished after %i steps. Total reward received is %f'
        % (length, agent.rewards))
  

if __name__ == '__main__':
  print("Hello")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=80,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=80,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str, default='tests/demo_map',
                      help='The environment level script to load')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script)