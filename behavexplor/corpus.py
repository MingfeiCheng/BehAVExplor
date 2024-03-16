import os
import copy
import pickle
import numpy as np

from loguru import logger

from behavexplor.element import CorpusElement
from behavexplor.models.coverage import CoverageModel

class InputCorpus(object):
    """Class that holds inputs and associated coverage."""
    def __init__(self, save_folder, window_size, cluster_num, threshold_coverage, feature_resample='linear'):

        # just for save
        self.window_size = window_size
        self.cluster_num = cluster_num
        self.threshold_coverage = threshold_coverage

        self.save_folder = save_folder
        self.coverage_model = CoverageModel(self.window_size,
                                            self.cluster_num,
                                            self.threshold_coverage)
        self.feature_resample = feature_resample

        self.corpus_total = [] # save all elements in the fuzzing

        self.corpus_seeds = [] # save seeds index
        self.corpus_fail = [] # save failed index
        self.corpus_coverage = [] # keep coverage

        self.parent_element = None

        self.current_selected_index = -1 # parent index
        self.current_mutation = None
        self.current_index = -1

    def _update_log(self, mutation_str):
        if mutation_str is not None:
            log_mutation = os.path.join(self.save_folder, 'logs/mutation.log')
            with open(log_mutation, 'a') as f:
                f.write(mutation_str)
                f.write('\n')

        corpus_str = 'total: {:d} seeds: {:d} fail: {:d} coverage: {:d}'.format(len(self.corpus_total),
                                                                                len(self.corpus_seeds),
                                                                                len(self.corpus_fail),
                                                                                len(self.corpus_coverage))
        log_corpus = os.path.join(self.save_folder, 'logs/corpus.log')
        with open(log_corpus, 'a') as f:
            f.write(corpus_str)
            f.write('\n')

        threshold = self.coverage_model.get_threshold_info()
        threshold_str = 'coverage: {:.5f}'.format(threshold['coverage'])
        log_threshold = os.path.join(self.save_folder, 'logs/model_threshold.log')
        with open(log_threshold, 'a') as f:
            f.write(threshold_str)
            f.write('\n')

        threshold_dyn = self.coverage_model.get_dynamic_threshold_info()
        threshold_dyn_str = 'coverage-dyn: {:.5f}'.format(threshold_dyn['coverage'])
        log_threshold_dyn = os.path.join(self.save_folder, 'logs/model_threshold_dyn.log')
        with open(log_threshold_dyn, 'a') as f:
            f.write(threshold_dyn_str)
            f.write('\n')

        model_static = self.coverage_model.get_static_info()
        model_static_str = 'coverage: {:d}'.format(model_static['coverage'])
        log_model_static = os.path.join(self.save_folder, 'logs/model_static.log')
        with open(log_model_static, 'a') as f:
            f.write(model_static_str)
            f.write('\n')

        logger.info(mutation_str)
        logger.info(corpus_str)
        logger.info(model_static_str)
        logger.info(threshold_str)
        logger.info(threshold_dyn_str)

    def _extract_element_feature(self, element):
        x, x_scale = element.get_vector()
        y_behavior, y_trace = self.coverage_model.extract_feature(x, x_scale, resample=self.feature_resample)
        element.set_feature_behavior(y_behavior)
        element.set_feature_trace(y_trace)
        return element

    def initialize(self, scenario_objs, scenario_recorders):
        # make sure inits have different routes.
        assert len(scenario_objs) == len(scenario_recorders)
        self.corpus_total = []  # save all elements in the fuzzing

        self.corpus_seeds = []  # save un-failed index
        self.corpus_fail = []  # save failed index
        self.corpus_coverage = []  # keep coverage

        self.parent_element = None
        self.current_index = 0

        init_size = len(scenario_objs)
        X_behavior = []

        for ele_index in range(init_size):
            scenario_obj = scenario_objs[ele_index]
            scenario_recorder = scenario_recorders[ele_index]
            corpus_element = CorpusElement(self.save_folder, scenario_obj, scenario_recorder)
            corpus_element = self._extract_element_feature(corpus_element)
            corpus_element.energy_init(0.5, 0.5)

            self.corpus_total.append(corpus_element)

            if corpus_element.is_fail():
                self.corpus_fail.append(self.current_index)
            else:
                # TODO: add energy
                self.corpus_seeds.append(self.current_index)
                self.corpus_coverage.append(self.current_index)
                # self._update_mutate_recorder(self.current_index, 0, False, False)
                # update coverage model
                x_behavior, x_trace = corpus_element.get_feature()
                X_behavior.append(x_behavior)
            self.current_index += 1

        # start if not zero
        if len(self.corpus_seeds) > 0:
            self.coverage_model.initialize(X_behavior)
        else:
            logger.warning('[Corpus] initialize with zero element.')

        self._update_log(None)
        return len(self.corpus_seeds)

    def sample_scenario_seed(self):

        if len(self.corpus_seeds) <= 0:
            logger.warning('[InputCorpus] length of corpus_seeds <= 0')
            return None, 'uniform'

        select_probabilities = []
        is_all_zero = True

        for i in range(len(self.corpus_seeds)):
            corpus_index = self.corpus_seeds[i]
            element = self.corpus_total[corpus_index]
            element_energy = element.get_energy()
            if element_energy > 0.0:
                is_all_zero = False
            select_probabilities.append(element_energy)

        if not is_all_zero:
            # compute new selection probability
            select_probabilities = np.array(select_probabilities)
            select_probabilities /= select_probabilities.sum()
            # add selection strategy
            corpus_index = np.random.choice(self.corpus_seeds, p=select_probabilities)
        else:
            corpus_index = np.random.choice(self.corpus_seeds)

        self.current_selected_index = corpus_index
        self.parent_element = copy.deepcopy(self.corpus_total[self.current_selected_index])

        sample_scenario_cfg = copy.deepcopy(self.parent_element.get_scenario_cfg())
        sample_seed_energy = self.parent_element.get_energy()

        self.current_mutation = 'uniform'
        if sample_seed_energy > 0.5:
            self.current_mutation = 'gauss'
        logger.info('[InputCorpus] Index: ' + str(self.current_selected_index) + ' Energy: ' + str(
            sample_seed_energy) + ' Mutation: ' + str(self.current_mutation))
        
        return sample_scenario_cfg, self.current_mutation
    
    def add_seed(self, scenario_obj, scenario_recorder):

        corpus_element = CorpusElement(self.save_folder, scenario_obj, scenario_recorder)
        corpus_element = self._extract_element_feature(corpus_element)

        is_fail = corpus_element.is_fail()
        is_novel = False

        corpus_feature_behavior, corpus_feature_trace = corpus_element.get_feature()
        feedback_coverage, min_coverage_distance, _ = self.coverage_model.feedback_coverage_behavior(corpus_feature_behavior, is_fail)

        corpus_element.set_feedback(feedback_coverage, min_coverage_distance)

        if is_fail:
            self.corpus_fail.append(self.current_index)
        else:
            if feedback_coverage or corpus_element.get_fitness() < self.parent_element.get_fitness():
                if feedback_coverage:
                    is_novel = True
                    self.corpus_coverage.append(self.current_index)
                # add to seed
                self.corpus_seeds.append(self.current_index)

        delta_coverage = min_coverage_distance
        delta_fitness = self.parent_element.get_fitness() - corpus_element.get_fitness()
        corpus_element.energy_init(delta_fitness, delta_coverage)

        # update parent energy
        if self.current_mutation == 'gauss':
            self.corpus_total[self.current_selected_index].energy_step(is_fail, is_novel, delta_fitness, delta_coverage, False)
        else:
            self.corpus_total[self.current_selected_index].energy_step(is_fail, is_novel, delta_fitness, delta_coverage, True)

        # parent step
        self.corpus_total.append(corpus_element)
        self.current_index += 1
        mutation_str = '(parent) energy: {:.5f} fitness: {:.5f} coverage-dist: {:.5f} energy-update: {:.5f}\n ' \
                       '(spring) energy: {:.5f} fitness: {:.5f} coverage-dist: {:.5f} \n'.format(
            self.parent_element.get_energy(), self.parent_element.get_fitness(), self.parent_element.get_coverage_distance(),
            self.corpus_total[self.current_selected_index].get_energy(),
            corpus_element.get_energy(), corpus_element.get_fitness(), corpus_element.get_coverage_distance()
        ) # before energy
        self._update_log(mutation_str)

        return len(self.corpus_seeds), len(self.corpus_fail)

    def __len__(self):
        return len(self.corpus_total)

    def save_models(self, save_folder=None):
        if save_folder is None:
            save_folder = os.path.join(self.save_folder, 'models')

        save_dict = {}
        for k, v in self.__dict__.items():
            if k == 'coverage_model':
                save_dict[k] = os.path.join(save_folder, 'coverage.pth')

        f = open(os.path.join(save_folder, 'corpus.pth'), 'wb')
        pickle.dump(save_dict, f, 2)
        f.close()
        # save layer
        self.coverage_model.save_model(save_folder)
