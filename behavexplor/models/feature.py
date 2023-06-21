import numpy as np

from tsfresh.feature_extraction import feature_calculators
from scipy import interpolate

def static_calculator(X):
    X_feature = []
    attribution_size = X.shape[1]
    for attr_i in range(attribution_size):
        attribution_i = X[:, attr_i]
        mean = feature_calculators.mean(attribution_i)
        minimum = feature_calculators.minimum(attribution_i)
        maximum = feature_calculators.maximum(attribution_i)
        mean_change = feature_calculators.mean_change(attribution_i)
        mean_abs_change = feature_calculators.mean_abs_change(attribution_i)
        variance = feature_calculators.variance(attribution_i)
        c3 = feature_calculators.c3(attribution_i, 1)
        cid_ce = feature_calculators.cid_ce(attribution_i, True)
        attribution_i_feature = [mean, variance, minimum, maximum, mean_change, mean_abs_change, c3, cid_ce]

        X_feature += attribution_i_feature

    return X_feature

class FeatureNet(object):

    def __init__(self, window_size=1):
        self.resample_frequency = 0
        self.window_size = window_size # unit is second (s)
        self.local_feature_extractor = None

    @staticmethod
    def input_select(x):
        # X: [T, N] T seg 0.1s
        # [time, speed, acc, ego_x, ego_y, ego_z, v_x, v_z, acc_x, acc_z, angular_x, angular_y, angular_z, rotation_x, rotation_y, rotation_z]
        trace_pos = np.concatenate([x[:, 0:1], x[:, 3:4], x[:, 5:6]], axis=1) # position
        # ->
        # [time, speed, acc, v_x, v_z, acc_x, acc_z, angular_y, rotation_y]
        behavior_vector = np.concatenate([x[:, 0:3], x[:, 6:10], x[:, 11:12], x[:, 14:15]], axis=1)
        return behavior_vector, trace_pos

    @staticmethod
    def input_resample(x, time_scale, resample='linear', sample_frequency=0.1):
        # [time, speed, acc, v_x, v_z, acc_x, acc_z, angular_y, rotation_y]
        assert resample in ['none', 'linear', 'quadratic', 'cubic']
        if resample == 'none':
            new_x = x[:, 1:]
        else:
            # X: [T, N] T seg 0.1s
            resample_axis = np.arange(time_scale[0], time_scale[1], sample_frequency)
            time_axis = x[:, 0]
            new_x = []
            for i in range(1, x.shape[1]):
                x_i = x[:, i]
                f_i = interpolate.interp1d(time_axis, x_i, kind=resample)
                new_x_i = f_i(resample_axis)
                new_x_i = np.append(new_x_i, x_i[-1])
                new_x.append(new_x_i)
            new_x = np.array(new_x)
            new_x = new_x.T
            # new_x:
            # [speed, acc, v_x, v_z, acc_x, acc_z, angular_y, rotation_y]
        return new_x

    def forward(self, x, x_scale, resample='linear'):
        x_behavior_vector, x_trace_pos = self.input_select(x)
        x_behavior_vector = self.input_resample(x_behavior_vector, x_scale, resample)
        x_trace_pos = self.input_resample(x_trace_pos, x_scale, resample)

        if self.window_size <= 0:
            return x_behavior_vector, x_trace_pos
        time_size = x_behavior_vector.shape[0]
        if time_size < self.window_size:
            last_element = x_behavior_vector[-1:, :]
            for _ in range(self.window_size - time_size):
                x_behavior_vector = np.concatenate([x_behavior_vector, last_element], axis=0)

        y = []
        for i in range(time_size - self.window_size + 1):
            x_segment = x_behavior_vector[i:i+self.window_size]
            x_feature = static_calculator(x_segment)
            y.append(x_feature)

        return np.array(y), x_trace_pos

