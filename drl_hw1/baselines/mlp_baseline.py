import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from drl_hw1.utils.gym_env import EnvSpec
from torch.utils.data import TensorDataset, DataLoader

class MLPBaseline:
    def __init__(self, env_spec: EnvSpec, hidden_sizes=(64,64), learning_rate=1e-4, epoch=10):
        self.feature_size = env_spec.observation_dim + 4
        self.loss_fn = nn.MSELoss(size_average=False)
        self.learning_rate = learning_rate
        self.hidden_sizes = hidden_sizes
        self.model = None
        self.epoch = epoch

    def _features(self, path):
        # compute regression features for the path
        o = np.clip(path["observations"], -10, 10)
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 1000.0
        feat = np.concatenate([o, al, al**2, al**3, np.ones((l, 1))], axis=1)
        return feat

    def fit(self, paths, return_errors=False):

        self.model = nn.Sequential()
        self.model.add_module('fc_0', nn.Linear(self.n, self.hidden_sizes[0]))
        self.model.add_module('tanh_0', nn.Tanh())
        self.model.add_module('fc_1', nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]))
        self.model.add_module('tanh_1', nn.Tanh())
        self.model.add_module('fc_2', nn.Linear(self.hidden_sizes[1], 1))


        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        dataset = TensorDataset(featmat, returns)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if return_errors:
            error_before = self.get_error(data_loader)

        for _ in range(self.epoch):
            for batch_idx, (data, target) in enumerate(data_loader):
                predictions = self.model(data)
                loss = self.loss_fn(predictions, target)
                error_before += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if return_errors:
            error_after = self.get_error(data_loader)
            return error_before, error_after

    def get_error(self, data_loader):
        error = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            predictions = self.model(data)
            error += self.loss_fn(predictions, target)
        return error

    def predict(self, path):
        if self.model is None:
            return np.zeros(len(path["rewards"]))
        return self.model(self._features(path))