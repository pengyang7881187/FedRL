"""The code is adapted from gym"""
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from MyEnv import MyEnv

tgt1 = '<geom axisangle="0 1 0 -3.8" name="bthigh" pos=".1 0 -.13" size="0.046 .145" type="capsule"/>'
tgt2 = '<geom axisangle="0 1 0 .52" name="fthigh" pos="-.07 0 -.12" size="0.046 .133" type="capsule"/>'
tgt_size = 'size'

tgt1_pos = 3707 + 7
tgt2_pos = 4594 + 7

# You need to modify this variable by yourself
site_packages_dir = '/miniconda3/envs/torch/lib/python3.8/'

asset_path = site_packages_dir + 'site-packages/gym/envs/mujoco/assets/'
ori_xml_path = asset_path + 'half_cheetah.xml'


def generate_xml(para):
    xml_file = 'half_cheetah' + ("%.2f" % para)[2:] + '.xml'
    xml_file_path = asset_path + xml_file
    if not os.path.exists(xml_file_path):
        with open(ori_xml_path, 'r') as f:
            # We modify the length of the thigh
            s = f.read()

        bthigh_length = round(143 + (para - 0.5) * 85)
        fthigh_length = round(133 + (para - 0.5) * 65)
        s_lst = list(s)
        s_lst[tgt1_pos: tgt1_pos + 3] = list(str(bthigh_length))
        s_lst[tgt2_pos: tgt2_pos + 3] = list(str(fthigh_length))
        s_new = ''.join(s_lst)

        with open(xml_file_path, 'w') as f_w:
            f_w.write(s_new)
    return xml_file


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class MyHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle, MyEnv):
    def __init__(self,
                 para=0.5,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        MyEnv.__init__(self, para)
        para = round(para, 2)
        xml_file = generate_xml(para)
        utils.EzPickle.__init__(**locals())

        self.time = 0

        self.embedding = np.zeros((100))

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost

        self.time += 1
        if self.time >= 1000:
            done = True
            self.time = 0
        else:
            done = False

        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return np.append(self.embedding, np.array(observation)), reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        self.time = 0

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return np.append(self.embedding, np.array(observation))

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
