import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, granularity=10, num_tasks=1):
        self.granularity = granularity
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE

        # granularity is the number of squares over the x axis, which is the same over the y-axis.
        granularity = self.granularity
        square_length = 20 / granularity
        idx = np.arange(0, granularity)

        if is_evaluation:
            # Pick the i-th square on the x axis (starting from the left)
            # Pick the j-th square on the y axis (starting from the bottom), so that i + j % 2 == 0
            x_rand_ind = np.random.choice(idx)
            y_rand_ind = np.random.choice(np.arange(x_rand_ind % 2, stop=granularity, step=2))
            assert (x_rand_ind + y_rand_ind) % 2 == 0
        else:
            x_rand_ind = np.random.choice(idx)
            y_rand_ind = np.random.choice(np.arange( (x_rand_ind % 2 == 0) * 1, stop=granularity, step=2))
            assert (x_rand_ind + y_rand_ind) % 2 == 1

        x = np.random.uniform(0, 1)
        x = -10 + (x_rand_ind + x) * square_length

        y = np.random.uniform(0, 1)
        y = -10 + (y_rand_ind + y) * square_length

        # x = np.random.uniform(-10, 10)
        # y = np.random.uniform(-10, 10)
        self._goal = np.array([x, y])

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
