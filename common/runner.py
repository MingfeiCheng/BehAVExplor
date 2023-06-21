import os
import pickle
import shutil

from loguru import logger

def clear_and_create(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

class Runner(object):

    def __init__(self, output_path, simulator):
        
        self.global_id = 0

        self.SCENARIO_FOLDER = 'scenarios'
        self.RECORD_LGSVL_FOLDER = 'records_lgsvl'
        self.RECORD_APOLLO_FOLDER = 'records_apollo'

        self.scenario_path = os.path.join(output_path, 'simulation', self.SCENARIO_FOLDER)
        self.record_lgsvl_path = os.path.join(output_path, 'simulation', self.RECORD_LGSVL_FOLDER)
        self.record_apollo_path = os.path.join(output_path, 'simulation', self.RECORD_APOLLO_FOLDER)

        clear_and_create(self.scenario_path)
        clear_and_create(self.record_lgsvl_path)
        clear_and_create(self.record_apollo_path) # self.record_path

        self.sim = simulator # save record to records/scenario_name/scenario_id
        
        self.runner_log = os.path.join(output_path, 'logs/runner.log')
        if os.path.exists(self.runner_log):
            os.remove(self.runner_log)

    def run(self, scenario_obj, scenario_id=None):

        if scenario_id is None:
            scenario_id = 'scenario_' + str(self.global_id)

        scenario_file = os.path.join(self.scenario_path, scenario_id + '.obj')
        with open(scenario_file, 'wb') as s_f:
            pickle.dump(scenario_obj, s_f)

        record_lgsvl_file = os.path.join(self.record_lgsvl_path, scenario_id + '.obj')

        if os.path.isfile(record_lgsvl_file):
            os.remove(record_lgsvl_file)

        while True:
            try:
                sim_recorder = self.sim.run(scenario_obj, scenario_id, self.record_apollo_path)
            except Exception as e:
                logger.error('Simulator may has some unexpected error:')
                logger.error('    ' + str(e))
            else:
                break

        # sim_recorder = self.sim.run(scenario_obj, scenario_id, self.record_apollo_path)

        with open(record_lgsvl_file, 'wb') as f:
            pickle.dump(sim_recorder, f)

        sim_events = sim_recorder.obtain_case_event_str()
        sim_fitness = sim_recorder.get_case_fitness()
        with open(self.runner_log, 'a') as f:
            f.write(str(scenario_id) + ' ')
            f.write(str(sim_events) + ' ')
            f.write(str(list(sim_fitness)))
            f.write('\n')

        self.global_id += 1

        logger.info('[Runner] Simulation Result: ')
        logger.info('[Runner]     Events: ' + str(sim_events))
        logger.info('[Runner]     Fitness: ' + str(list(sim_fitness)))

        return sim_recorder, scenario_id

    def close(self):
        self.sim.close()