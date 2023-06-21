import copy

import numpy as np

class CorpusElement(object):
    """Class representing a single element of a corpus."""

    def __init__(self, save_folder, scenario_cfg, scenario_recorder, init_energy=1.0):
        self.parent_id = None
        self.feature_behavior = None
        self.feature_trace = None
        self.scenario_vector = None
        self.scenario_time_scale = None

        self.energy = init_energy
        self.energy_recorder = {
            'fail': 0,
            'novel': 0,
            'valid': 0,
            'total': 0,
        }

        self.save_folder = save_folder
        self.scenario_id = scenario_recorder.get_case_id()
        self.scenario_cfg = scenario_cfg
        self.scenario_recorder = scenario_recorder

        self.fitness = None
        self.feedback_coverage = False # Bool - true or false
        self.min_coverage_distance = np.inf

    def get_parent_id(self):
        return self.parent_id

    def get_scenario_id(self):
        return self.scenario_id

    def get_scenario_cfg(self):
        return copy.deepcopy(self.scenario_cfg)

    def get_scenario_recorder(self):
        return copy.deepcopy(self.scenario_recorder)

    def get_events(self):
        case_events = self.scenario_recorder.get_case_event()
        return case_events

    def get_vector(self):
        if self.scenario_vector is None or self.scenario_time_scale is None:
            self.scenario_vector = self.scenario_recorder.get_case_vector()
            self.scenario_time_scale = self.scenario_recorder.get_time_scale()
        return self.scenario_vector, self.scenario_time_scale

    def get_feature(self):
        return self.feature_behavior, self.feature_trace

    def get_energy(self):
        if self.energy < 0.0:
            self.energy = 0.0
        return self.energy

    def get_fitness(self):
        case_fitness = self.scenario_recorder.get_case_fitness()
        obj_dest = 1.0 - min(case_fitness[0], 10) / 10.0  # may need to be deleted 0 better
        obj_collision = min(case_fitness[1], 1.0)  # min(all_fitness[1], 1) 0 better
        obj_yellow_line = min(case_fitness[2], 1.0)  # min(all_fitness[2], 1) 0 better
        obj_edge_line = min(case_fitness[3], 1.0)  # min(all_fitness[3], 1) 0 better
        self.fitness = (obj_dest + obj_collision + obj_yellow_line + obj_edge_line) / 4.0
        return self.fitness

    def get_scenario_topology(self):
        return self.scenario_cfg.get_scenario_structure()

    def get_coverage_distance(self):
        return self.min_coverage_distance

    def set_scenario_cfg(self, scenario_cfg):
        self.scenario_cfg = scenario_cfg

    def set_parent_id(self, parent_id):
        self.parent_id = parent_id

    def set_feature_behavior(self, x):
        self.feature_behavior = x

    def set_feature_trace(self, x):
        self.feature_trace = x

    def set_feedback(self, feedback, distance):
        self.feedback_coverage = feedback
        self.min_coverage_distance = distance

    def energy_step(self, is_fail, is_novel, delta_fitness, delta_coverage, is_uniform):
        """
        for new added element:
            main part is delta
        delta_fitness: -1 - 1, larger better
            spring - parent, low value means better
        delta_coverage: 0 - 1, larger better
        for parent element:
            main part is the state of springs, reduce their probability of
        """
        benign = True
        if is_uniform:
            self.energy = self.energy - 0.1
        else:
            self.energy_recorder['total'] += 1
            if is_fail or is_novel:
                benign = False
                self.energy_recorder['valid'] += 1
                if is_fail:
                    self.energy_recorder['fail'] += 1
                if is_novel:
                    self.energy_recorder['novel'] += 1

            valid_rate = self.energy_recorder['valid'] / float(self.energy_recorder['total'])

            if benign:
                ef = -0.1 * (1 - valid_rate)
            else:
                ef = valid_rate

            self.energy = self.energy + 0.5 * ef + 0.5 * np.tanh(delta_fitness / (1 - delta_coverage + 1e-10)) - 0.05 * 1

    def energy_init(self, delta_fitness, delta_coverage):
        """
        delta_fitness: [0, 1] 0 is better
        delta_coverage: [0, 1] 1 is better
        """
        # delta part max limit to 2
        # delta_part = (1.0 - delta_fitness) + delta_coverage
        delta_part = delta_fitness / 2.0 + delta_coverage / 2.0
        self.energy += delta_part

    def is_fail(self):
        fail_lst = [1, 3, 4, 6, 7, 14]
        case_events = self.scenario_recorder.get_case_event()
        for fail_item in fail_lst:
            if fail_item in case_events:
                return True
        return False
