import os
import math
import time
import numpy as np
import gym
import pybullet as p
import pybullet_data
from .balancebot import BalanceBot
from gym import spaces

class BalancebotEnv(gym.Env):

    def __init__(self, 
                render=False):

        self._urdfRoot = pybullet_data.getDataPath()
        self._time_step = 0.002
        self._control_latency = self._time_step * 1.0
        self._action_repeat = 1
        
        self.vis_time_step = 0.002
        self._is_render = render
        self._last_frame_time = 0.0
        self.total_step = 0
        self.prev_action = 0

        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
      
        if (render):
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

    def _getObservation(self):

        row, pitch, _ = self.balancebot.getBaseRollPitchYaw()
        vel, vel2     = self.balancebot.getMotorStrength()
        
        row = (row / 1.5707963) 
        pitch = (pitch / 1.5707963)

        self._observation[7] = self._observation[5]
        self._observation[6] = self._observation[4]
        self._observation[5] = self.prev_action 
        self._observation[4] = self.prev_action 

        
        self._observation[3] = self._observation[1]
        self._observation[2] = self._observation[0]

        self._observation[1] = pitch
        self._observation[0] = row 
      
        observation = np.array(self._observation).flatten()
        return observation

    def step(self, action):

        assert type(action) == np.ndarray

        if self._is_render:
            
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.

            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.vis_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        self.balancebot.step(action)
        self.prev_action = action
        observation = self._getObservation()
        self.total_step += 1
        
        return observation, self._reward(), self._terminal(), {}

    def reset(self):
        
        p.resetSimulation()
        p.setGravity(0,0,-10)
        p.setTimeStep(self._time_step)

        plane = p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"), [0, 0, 0])

        self._observation = [0, 0, 0, 0, 0, 0, 0, 0]
        self._objectives = []

        self.balancebot = BalanceBot(
            pybullet_client=p, 
            action_repeat=self._action_repeat, 
            time_step=self._time_step,
            control_latency=self._control_latency)

        self.balancebot.reset()
        observation = self._getObservation()
        return observation


    def _reward(self):

        curr_p, curr_v, curr_a, curr_w = self.balancebot._observation_history[0]
        prev_p, prev_v, prev_a, prev_w = self.balancebot._observation_history[1]

        cx, cy, cz = curr_p
        cr, cp, cy = curr_a
        cvx, cvy, cvz = curr_v
        cwr, cwp, cwy = curr_w
        
        px, py, pz = prev_p
        pr, pp, py = prev_a
        pvx, pvy, pvz = prev_v
        pwr, pwp, pwy = prev_w
        
        
        ang_reward = 1.0 - abs(cr+pr)/2.0
        ang_vel_reward = abs(cwr+pwr)/2.0

        pos_reward = abs(cx+px)/2.0 + abs(cy+py)/2.0
        pos_vel_reward = abs(cvx+pvx)/2.0 + abs(cvy+pvy)/2.0
        

        self._objectives.append([pos_reward, pos_vel_reward, ang_vel_reward, ang_reward])
        
        return -0.00*pos_reward -0.01*pos_vel_reward -0.00*ang_vel_reward  +1.00*ang_reward

    def get_objectives(self):
        return self._objectives
        
    def _terminal(self):
        
        ang = self._observation[0]
        critiria = 1.0
        
        if abs(ang) > critiria:
            return True
        else:
            return False


    def render(self, mode='human', close=False):
        return None

    def close(self):
        return None


