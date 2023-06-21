import copy
import random
import numpy as np

from loguru import logger

from common import utils

NPC_AGENT_TYPES = ["Sedan", "SUV", "Jeep", "Hatchback", "SchoolBus"]

offset_fix_value = 2.5
min_speed_limit = 0.2

class Scenario(object):

    def __init__(self):
        self.ego = None # dict

        # init npc para
        self.waypoint_num = None
        self.npc_size = None
        self.npc_types = []
        self.npc_routes = []
        self.npc_routes_ids = []
        self.npc_waypoints = [] #[[[]]]

        # other oracle
        self.lanes = None
        self.yellow_lines = [] # [[line_info]]
        self.edge_lines = [] # [[line_info]]
        self.cross_lines = [] # [[line_info]]
        self.routes = [] # ids
        self.route_details = []

        # environment
        self.environment = None

        # offset recorder
        self.ego_offset_recorder = None
        self.current_offset_recorder = None

    def generate_specific_info_wo_npc(self, basic_info, route_info):

        lane_details = route_info['lane_details']
        # obtain legal_lines
        ego_start_lane_id = basic_info['ego']['start']['lane_id']
        ego_dest_lane_id = basic_info['ego']['destination']['lane_id']

        ego_lanes_in = [ego_start_lane_id]
        ego_lanes_out = [ego_dest_lane_id]
        ego_lanes_mid = []

        lane_data = lane_details[ego_start_lane_id]
        for item in lane_data['right_neighbor_lane']:
            if item['id'] not in lane_details.keys():
                continue
            ego_lanes_in.append(item['id'])
        for item in lane_data['left_neighbor_lane']:
            if item['id'] not in lane_details.keys():
                continue
            ego_lanes_in.append(item['id'])

        lane_data = lane_details[ego_dest_lane_id]
        for item in lane_data['right_neighbor_lane']:
            if item['id'] not in lane_details.keys():
                continue
            ego_lanes_out.append(item['id'])
        for item in lane_data['left_neighbor_lane']:
            if item['id'] not in lane_details.keys():
                continue
            ego_lanes_out.append(item['id'])

        for lane_id in ego_lanes_in:
            # TODO: need check
            if lane_id not in lane_details.keys():
                continue
            successors = lane_details[lane_id]['successor']
            for j in range(len(successors)):
                successor_j = successors[j]['id']
                if successor_j not in lane_details.keys():
                    continue
                successor_j_data = lane_details[successor_j]
                for j_s in successor_j_data['successor']:
                    if j_s['id'] in ego_lanes_out:
                        ego_lanes_mid.append(successor_j)

        ego_lanes = ego_lanes_in
        ego_lanes += ego_lanes_out
        ego_lanes += ego_lanes_mid
        ego_lanes = list(set(ego_lanes))
        logger.info('Ego lanes: ' + str(ego_lanes))

        for lane_id in ego_lanes:
            lane_data = lane_details[lane_id]
            lane_turn = lane_data['turn']
            # turn: NO_TURN/RIGHT_TURN/LEFT_TURN
            # left boundary
            left_boundary = lane_data['left_boundary']
            left_boundary_type = left_boundary['type'][0]
            left_boundary_points = left_boundary['points']
            if left_boundary_type == 'CURB':
                self.edge_lines.append(left_boundary_points)
            elif left_boundary_type == 'DOUBLE_YELLOW':
                self.yellow_lines.append(left_boundary_points)
            elif left_boundary_type == 'DOTTED_WHITE':
                if lane_id not in ego_lanes_mid:
                    self.cross_lines.append(left_boundary_points)
            else:
                raise RuntimeError('No lane type: ' + str(left_boundary_type))

            # right boundary
            right_boundary = lane_data['right_boundary']
            right_boundary_type = right_boundary['type'][0]
            right_boundary_points = right_boundary['points']
            if right_boundary_type == 'CURB':
                self.edge_lines.append(right_boundary_points)
            elif right_boundary_type == 'DOUBLE_YELLOW':
                self.yellow_lines.append(right_boundary_points)
            elif right_boundary_type == 'DOTTED_WHITE':
                if lane_id not in ego_lanes_mid:
                    self.cross_lines.append(right_boundary_points)
            else:
                raise RuntimeError('No lane type: ' + str(right_boundary_type))

        self.ego = basic_info['ego']
        self.lanes = route_info['lane_details']
        self.route_details = route_info['route_details']
        self.routes = route_info['routes'] # route id
        self.waypoint_num = basic_info['npcs']['waypoint']
        self.npc_size = basic_info['npcs']['npc_num']
        self.ego_offset_recorder = {self.ego['start']['lane_id']: [
            [[offset_fix_value, self.lanes[self.ego['start']['lane_id']]['central']['length'] - offset_fix_value]],
            [[-200, self.ego['start']['offset'] - 15], [self.ego['start']['offset'] + 15, 9999]]
        ]}
        self.environment = basic_info['environment']
        logger.info('[Scenario] OK - Generate specific scenario wo NPCs.')

    def mutation_route(self):
        offset_recorder = copy.deepcopy(self.ego_offset_recorder)
        for i in range(self.npc_size):
            if random.random() > 0.5:
                self.npc_types[i] = random.choice(NPC_AGENT_TYPES)
                self.npc_routes_ids[i] = random.choice(self.routes)
                self.npc_routes[i] = self.route_details[self.npc_routes_ids[i]]

                npc_i_route = self.npc_routes[i]
                npc_i_offset_speed = []
                for _ in range(len(npc_i_route)):
                    for _ in range(self.waypoint_num):
                        npc_i_offset_speed.append([])
                npc_i_offset_speed, offset_recorder = self._uniform_mutation(npc_i_route, offset_recorder)
                self.npc_waypoints[i] = npc_i_offset_speed

    def generate_random_abstract_scenario(self):
        self.npc_routes = []
        self.npc_types = []
        self.npc_routes_ids = []
        for i in range(self.npc_size):
            npc_i_type = random.choice(NPC_AGENT_TYPES)
            self.npc_types.append(npc_i_type)

            npc_i_route_id = random.choice(self.routes)
            self.npc_routes_ids.append(npc_i_route_id)

            npc_i_route = self.route_details[npc_i_route_id]
            self.npc_routes.append(npc_i_route)
        # logger.info('[Scenario] OK - Generate random npc configure.')

    def generate_random_concrete_scenario(self):
        self.npc_waypoints = []
        offset_recorder = copy.deepcopy(self.ego_offset_recorder)

        for i in range(self.npc_size):
            npc_i_route = self.npc_routes[i]
            npc_i_offset_speed = []
            for _ in range(len(npc_i_route)):
                for _ in range(self.waypoint_num):
                    npc_i_offset_speed.append([])

            npc_i_offset_speed, offset_recorder = self._uniform_mutation(npc_i_route, offset_recorder)
            self.npc_waypoints.append(npc_i_offset_speed)

    def mutation_uniform(self):
        # mutate_npc_types = copy.deepcopy(self.npc_types)
        mutate_npc_routes = copy.deepcopy(self.npc_routes)
        mutate_npc_routes_ids = copy.deepcopy(self.npc_routes_ids)
        mutate_npc_offset_speed = copy.deepcopy(self.npc_waypoints)
        offset_recorder = copy.deepcopy(self.ego_offset_recorder)

        # mutate structure & type - drop
        # adjust offset & speed
        for npc_id in range(self.npc_size):
            npc_i_route = mutate_npc_routes[npc_id]
            npc_i_offset_speed, offset_recorder = self._uniform_mutation(npc_i_route, offset_recorder)
            mutate_npc_offset_speed[npc_id] = npc_i_offset_speed

        self.npc_routes = mutate_npc_routes
        self.npc_routes_ids = mutate_npc_routes_ids
        self.npc_waypoints = mutate_npc_offset_speed
        logger.info('[Scenario] OK - Generate mutated NPC configure with mutation_normal')

    def mutation_gauss(self, offset_mutation, speed_mutation):
        # mutate_npc_types = copy.deepcopy(self.npc_types)
        mutate_npc_routes = copy.deepcopy(self.npc_routes)
        mutate_npc_routes_ids = copy.deepcopy(self.npc_routes_ids)
        mutate_npc_offset_speed = copy.deepcopy(self.npc_waypoints)
        offset_recorder = copy.deepcopy(self.ego_offset_recorder)

        # mutate structure & type - drop
        # adjust offset & speed
        for npc_id in range(self.npc_size):
            npc_i_route = mutate_npc_routes[npc_id]
            npc_i_offset_speed = mutate_npc_offset_speed[npc_id]
            npc_i_offset_speed, offset_recorder = self._gauss_mutation(npc_i_route, npc_i_offset_speed, offset_recorder, offset_mutation, speed_mutation)
            mutate_npc_offset_speed[npc_id] = npc_i_offset_speed

        self.npc_routes = mutate_npc_routes
        self.npc_routes_ids = mutate_npc_routes_ids
        self.npc_waypoints = mutate_npc_offset_speed
        logger.info('[Scenario] OK - Generate mutated NPC configure with mutation_normal')

    def _gauss_mutation(self, npc_i_route, npc_i_offset_speed, offset_recorder, offset_mutation, speed_mutation):
        last_lane_id = None
        last_offset = None
        last_trace_length = 0.0
        # TODO:Fix not the same length of speed & offset
        wp_length = len(npc_i_route) * self.waypoint_num
        assert wp_length == len(npc_i_offset_speed)

        for wp_i in range(len(npc_i_offset_speed)):

            current_offset = npc_i_offset_speed[wp_i][0]
            current_speed = npc_i_offset_speed[wp_i][1]
            
            wp_i_lane_id_index = wp_i // self.waypoint_num

            wp_i_lane_id = npc_i_route[wp_i_lane_id_index]
            wp_i_lane_length = self.lanes[wp_i_lane_id]['central']['length']
            wp_i_lane_speed_limit = self.lanes[wp_i_lane_id]['speed_limit']

            if random.random() < 0.5:
                mutate_speed = current_speed
                mutate_offset = current_offset
            else:
                mutate_speed = random.gauss(current_speed, speed_mutation)
                mutate_offset = random.gauss(current_offset, offset_mutation)

            if wp_i == 0:  # start
                mutate_speed = float(np.clip(mutate_speed, min_speed_limit, wp_i_lane_speed_limit))
                mutate_offset, offset_recorder = utils.check_relocate(offset_recorder, mutate_offset,
                                                                      wp_i_lane_id, offset_fix_value,
                                                                      wp_i_lane_length - offset_fix_value)
            elif wp_i == len(npc_i_offset_speed) - 1:  # destination
                mutate_speed = 0.0
                if wp_i_lane_id == last_lane_id:
                    mutate_offset = float(np.clip(mutate_offset, last_offset, wp_i_lane_length + last_trace_length))
                else:
                    last_trace_length += self.lanes[last_lane_id]['central']['length']
                    mutate_offset = float(np.clip(mutate_offset, last_trace_length, wp_i_lane_length + last_trace_length))
            else:
                mutate_speed = float(np.clip(mutate_speed, min_speed_limit, wp_i_lane_speed_limit))
                if wp_i_lane_id == last_lane_id:
                    mutate_offset = float(np.clip(mutate_offset, last_offset, wp_i_lane_length + last_trace_length))
                else:
                    last_trace_length += self.lanes[last_lane_id]['central']['length']
                    mutate_offset = float(np.clip(mutate_offset, last_trace_length, wp_i_lane_length + last_trace_length))

            npc_i_offset_speed[wp_i][0] = mutate_offset
            npc_i_offset_speed[wp_i][1] = mutate_speed

            last_lane_id = wp_i_lane_id
            last_offset = mutate_offset
        return npc_i_offset_speed, offset_recorder

    def _uniform_mutation(self, npc_i_route, offset_recorder):

        npc_i_offset_speed = []
        for _ in range(len(npc_i_route)):
            for _ in range(self.waypoint_num):
                npc_i_offset_speed.append([])

        waypoint_id = 0
        last_trace_length = 0.0
        last_offset = None
        last_lane_id = None
        for route_index in range(len(npc_i_route)):
            route_lane_id = npc_i_route[route_index]
            lane_data = self.lanes[route_lane_id]
            lane_length = lane_data['central']['length']
            lane_speed_limit = lane_data['speed_limit']

            for w_i in range(self.waypoint_num):
                if waypoint_id == 0:  # start lane & offset & speed
                    delta_start_offset = random.uniform(offset_fix_value, lane_length - offset_fix_value)
                    npc_offset, offset_recorder = utils.check_relocate(offset_recorder, delta_start_offset,
                                                                       route_lane_id, offset_fix_value,
                                                                       lane_length - offset_fix_value)
                    npc_speed = random.uniform(min_speed_limit, lane_speed_limit)
                elif waypoint_id == len(npc_i_offset_speed) - 1:  # end lane & offset & speed
                    npc_speed = 0.0
                    if route_lane_id == last_lane_id:
                        npc_offset = random.uniform(last_offset, lane_length + last_trace_length)
                    else:
                        last_trace_length += self.lanes[last_lane_id]['central']['length']
                        npc_offset = random.uniform(last_trace_length, lane_length + last_trace_length)
                else:
                    npc_speed = random.uniform(min_speed_limit, lane_speed_limit)
                    if route_lane_id == last_lane_id:
                        npc_offset = random.uniform(last_offset, lane_length + last_trace_length)
                    else:
                        last_trace_length += self.lanes[last_lane_id]['central']['length']
                        npc_offset = random.uniform(last_trace_length, lane_length + last_trace_length)

                last_offset = npc_offset

                npc_i_offset_speed[waypoint_id] = [npc_offset, npc_speed]
                waypoint_id += 1

                last_lane_id = route_lane_id

        assert waypoint_id == len(npc_i_offset_speed)

        return npc_i_offset_speed, offset_recorder

    def get_yellow_lines(self):
        return self.yellow_lines

    def get_cross_lines(self):
        return self.cross_lines

    def get_edge_lines(self):
        return self.edge_lines

    def get_ego(self):
        return self.ego

    def get_npcs(self):
        return copy.deepcopy(self.npc_types), copy.deepcopy(self.npc_routes_ids), copy.deepcopy(self.npc_routes), copy.deepcopy(self.npc_waypoints)
    
    def get_scenario_structure(self):
        return copy.deepcopy(self.npc_routes_ids)

    def get_route_list(self):
        return self.routes

    def get_npc_num(self):
        return self.npc_size

    def get_lgsvl_input(self):
        lgsvl_input = {'ego': self.ego,
                       'ego_id': self.ego['agent_type'],
                       'npc_types': self.npc_types,
                       'npc_waypoints': self.npc_waypoints,
                       'npc_lanes_to_follow': self.npc_routes,
                       'npc_size': self.npc_size,
                       'lanes': self.lanes,
                       'yellow_lines': self.yellow_lines,
                       'cross_lines': self.cross_lines,
                       'edge_lines': self.edge_lines,
                       'environment': self.environment
                       }

        return copy.deepcopy(lgsvl_input)
