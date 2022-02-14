"""The code is adapted from gym"""
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
from MyEnv import MyEnv

tgt1 = '<geom friction="0.9" fromto="0 0 1.05 0 0 0.6" name="thigh_geom" size="0.05" type="capsule"/>'
tgt2 = '<geom friction="0.9" fromto="0 0 0.6 0 0 0.1" name="leg_geom" size="0.04" type="capsule"/>'
tgt3 = '<geom friction="2.0" fromto="-0.13 0 0.1 0.26 0 0.1" name="foot_geom" size="0.06" type="capsule"/>'
tgt_size = 'size'
tgt1_pos = 1537
tgt2_pos = 1771
tgt3_pos = 2025

# You need to modify this variable by yourself
site_packages_dir = '/miniconda3/envs/torch/lib/python3.8/'

asset_path = site_packages_dir + 'site-packages/gym/envs/mujoco/assets/'
ori_xml_path = asset_path + 'hopper.xml'


def generate_xml(para):
    xml_file = 'hopper' + ("%.2f" % para)[2:] + '.xml'
    xml_file_path = asset_path + xml_file
    if not os.path.exists(xml_file_path):
        with open(ori_xml_path, 'r') as f:
            # We modify the length of the leg
            s = f.read()

        pole_length = "%.3f" % (0.04 + (para - 0.5) * 0.02)
        new_s = s[:tgt2_pos + 4] + ' ' + s[tgt2_pos + 4:]
        s_lst = list(new_s)
        s_lst[tgt2_pos: tgt2_pos + 5] = list(str(pole_length))
        s_new = ''.join(s_lst)

        with open(xml_file_path, 'w') as f_w:
            f_w.write(s_new)
    return xml_file


class MyHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle, MyEnv):
    def __init__(self, para=0.5):
        MyEnv.__init__(self, para)
        # We assume para lives in (0, 1)
        para = round(para, 2)
        xml_file = generate_xml(para)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
