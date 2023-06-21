import os
import sys
import yaml
import json
import copy
import random
import datetime
import argparse

from time import strftime, gmtime

from loguru import logger
from behavexplor.utils import create_path
from behavexplor.corpus import InputCorpus

from common.runner import Runner
from common.simulator import Simulator
from common.scenario import Scenario

level = "INFO"
logger.configure(handlers=[{"sink": sys.stderr, "level": level}]) # TODO: fix file output

class Fuzzer(object):

    def __init__(self, cfgs):
        
        now = datetime.datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self.cfgs = cfgs
        self.cfgs['output_path'] = self.cfgs['output_path'] + '-at-' + date_time
        self.output_path = self.cfgs['output_path']

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        create_path(self.output_path)
        self.sim = Simulator(self.cfgs['max_sim_time'], self.cfgs['lgsvl_map'], self.cfgs['apollo_map'], sim_mode=self.cfgs['sim_mode'])

    def loop(self, time_limitation):

        scenario_basic_file = self.cfgs['scenario_basic_file']
        scenario_route_file = self.cfgs['scenario_route_file']

        with open(scenario_basic_file, 'r') as f:
            scenario_basic_info = json.load(f)
        with open(scenario_route_file, 'r') as f:
            scenario_route_info = json.load(f)

        if self.cfgs['ego_behavior'] > len(scenario_basic_info) - 1:
            raise RuntimeError('Max ego_behavior is ' + str(len(scenario_basic_info) - 1))

        scenario_basic_info = scenario_basic_info[self.cfgs['ego_behavior']]
        ego_behavior = str(scenario_basic_info['name'])
        logger.info('[Fuzzer] Ego Behavior: ' + ego_behavior)

        log_file = os.path.join(self.output_path, 'logs/system.log')
        if os.path.exists(log_file):
            os.remove(log_file)
        logger.add(log_file, level=level)
        self.record_cfgs()

        runner = Runner(self.output_path, self.sim)
        input_corpus = InputCorpus(self.output_path,
                                   self.cfgs['window_size'],
                                   self.cfgs['cluster_num'],
                                   self.cfgs['threshold_coverage'],
                                   self.cfgs['feature_resample']
                                   )

        logger.info('[Fuzzer] Start fuzzer iteration. Total time: ' + str(strftime('%H:%M:%S', gmtime(int(time_limitation)))))

        init_scenario_obj = Scenario()
        init_scenario_obj.generate_specific_info_wo_npc(scenario_basic_info, scenario_route_info)
        init_scenario_lst = []
        for i in range(self.cfgs['init_seed_size']):
            scenario_obj = copy.deepcopy(init_scenario_obj)
            scenario_obj.generate_random_abstract_scenario()
            scenario_obj.generate_random_concrete_scenario()
            init_scenario_lst.append(scenario_obj)

        init_scenario_recorder_lst = []
        for i in range(len(init_scenario_lst)):
            scenario_obj = init_scenario_lst[i]
            scenario_recorder, scenario_id = runner.run(scenario_obj)
            # TODO: add latency detection & rerun
            init_scenario_recorder_lst.append(scenario_recorder)

        corpus_seed_size = 0
        while corpus_seed_size <= 0:
            corpus_seed_size = input_corpus.initialize(init_scenario_lst,
                                                       init_scenario_recorder_lst)
            if corpus_seed_size <= 0:
                logger.warning('[Fuzzer] Init seeds are all failed, generate one new element.')
                scenario_obj = copy.deepcopy(random.choice(init_scenario_lst))
                scenario_obj.mutation_uniform()
                scenario_recorder, scenario_id = runner.run(scenario_obj)

                init_scenario_lst = [scenario_obj]
                init_scenario_recorder_lst = [scenario_recorder]

        start_time = datetime.datetime.now()
        iteration = 0

        while True:
            logger.info('[Fuzzer] Iterations: ' + str(iteration))
            scenario_obj, mutation_method = input_corpus.sample_scenario_seed()

            for child_i in range(self.cfgs['child_size']):
                if mutation_method == 'uniform':
                    if scenario_obj is not None:
                        scenario_obj.generate_random_abstract_scenario()
                        scenario_obj.generate_random_concrete_scenario()
                    else:
                        scenario_obj = copy.deepcopy(init_scenario_obj)
                        scenario_obj.mutation_route()
                else:
                    scenario_obj.mutation_gauss(self.cfgs['offset_mutation'], self.cfgs['speed_mutation'])

                scenario_recorder, scenario_id = runner.run(scenario_obj)

                corpus_seed_size, corpus_fail_size = input_corpus.add_seed(scenario_obj, scenario_recorder)
                logger.info('[Fuzzer] corpus_seed_size: ' + str(corpus_seed_size) + ' corpus_fail_size: ' + str(corpus_fail_size))

                exec_time = int((datetime.datetime.now() - start_time).seconds)
                if exec_time >= time_limitation:
                    logger.info(
                        '[Fuzzer] Finish testing. Real execution time: ' + str(strftime('%H:%M:%S', gmtime(exec_time))))
                    input_corpus.save_models()
                    self.sim.close()
                    exit(-1)

            input_corpus.save_models()
            iteration += 1

    def record_cfgs(self):
        logger.info('[Fuzzer] Record fuzzer configs:')
        for k, v in self.cfgs.items():
            logger.info(str(k) + ' : ' + str(v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apollo Coverage Fuzzer Testing.')
    parser.add_argument('--config', type=str, help='Test config yaml file.', required=True)
    args = parser.parse_args()

    yaml_file = args.config
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    fuzzer = Fuzzer(params)
    fuzzer.loop(int(params['total_test_time'])) # seconds