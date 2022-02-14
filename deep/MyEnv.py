import gym


class MyEnv(gym.Env):
    def __init__(self, para, **kwargs):
        super(MyEnv, self).__init__(**kwargs)
        self.para = para
