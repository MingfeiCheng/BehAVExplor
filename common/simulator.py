import os
import lgsvl
import time
import numpy as np

from datetime import datetime
from loguru import logger

from common import utils
from common.frame import CaseRecorder, FrameElement, FrameEventType, CaseFaultType


class Simulator(object):

    def __init__(self, total_sim_time, lgsvl_map='SanFrancisco_correct', apollo_map='SanFrancisco', sim_mode='async'):

        self.sim = None
        self.ego = None
        self.destination = None
        # self.dv_mode = 'Mkz Lgsvl'
        self.async_flag = sim_mode
        # if sim_mode == 'async':
        #     self.async_flag = True
        # else:
        #     self.async_flag = False
        self.total_sim_time = total_sim_time
        self.lgsvl_map = lgsvl_map
        self.apollo_map = apollo_map

        self.mutated_npc_list = []
        self.yellow_lines = []
        self.cross_lines = []
        self.edge_lines = []

        self.connect_lgsvl()
        self.load_map(self.lgsvl_map)

        self.simulation_count = 0

        self.modules = [
            'Localization',  # ok
            'Transform',  # ok
            'Routing',
            'Prediction',  # ok
            'Planning',  # ok
            'Control',
            'Storytelling'  # ok
        ]

    def connect_lgsvl(self):
        try:
            sim = lgsvl.Simulator(address="127.0.0.1", port=8181)
            self.sim = sim
        except Exception as e:
            logger.error('[Simulator] Connect LGSVL wrong: ' + '127.0.0.1:8181')
            logger.error(e.message)
        logger.info('[Simulator] Connected LGSVL 127.0.0.1:8181')

    def load_map(self, mapName="SanFrancisco_correct"):
        if self.sim.current_scene == mapName:
            self.sim.reset()
        else:
            self.sim.load(mapName)
        logger.info('[Simulator] Loaded map: ' + mapName)

    def convert_coordinate(self, point):

        if self.sim is None:
            raise RuntimeError('[Simulator] sim is not be initialized.')
        converted_coord = self.sim.map_from_gps(northing=point['y'], easting=point['x'])
        return converted_coord

    def get_lane_start_end(self, lane_info, offset):
        points = lane_info['central']['points']
        if len(points) > 2:
            for i in range(1, len(points)):
                seg_end = points[i]
                seg_start = points[i - 1]
                seg_len = np.sqrt((seg_end['x'] - seg_start['x']) ** 2 + (seg_end['y'] - seg_start['y']) ** 2)
                if seg_len > offset:
                    return self.convert_coordinate(point=seg_start), self.convert_coordinate(
                        point=seg_end), offset, seg_len
                else:
                    offset = offset - seg_len
        else:
            return self.convert_coordinate(point=points[0]), self.convert_coordinate(point=points[-1]), offset, \
                   lane_info['central']['length']

    def convert_legal_lines(self, lines):
        convert_lines = []
        for i in range(len(lines)):
            pre_line = lines[i]
            after_line = []
            for j in range(len(pre_line)):
                line_point = self.convert_coordinate(pre_line[j])
                after_line.append([line_point.position.x, line_point.position.z])
            convert_lines.append(after_line)
        return convert_lines

    def init_environment(self, scenario_obj):
        lgsvl_input = scenario_obj.get_lgsvl_input()

        # load ego car
        ego_data = lgsvl_input['ego']
        ego_ID = lgsvl_input['ego_id']
        ego_start_lane_id = ego_data['start']['lane_id']
        ego_start_offset = ego_data['start']['offset']
        ego_start_lane_data = lgsvl_input['lanes'][ego_start_lane_id]

        lane_start, lane_end, ego_start_offset, lane_length = self.get_lane_start_end(ego_start_lane_data,
                                                                                      ego_start_offset)
        ego_start_position = utils.compute_start_position(lane_start=lane_start, lane_end=lane_end,
                                                          lane_length=lane_length, offset=ego_start_offset)

        ego_pos_vector = lgsvl.Vector(x=ego_start_position['x'], y=10.2, z=ego_start_position['z'])
        ego_state = lgsvl.AgentState()
        ego_state.transform = self.sim.map_point_on_lane(ego_pos_vector)
        self.ego = self.sim.add_agent(ego_ID, lgsvl.AgentType.EGO, ego_state)
        ## ego destination
        ego_dest_lane_id = ego_data['destination']['lane_id']
        ego_dest_offset = ego_data['destination']['offset']
        ego_dest_lane_data = lgsvl_input['lanes'][ego_dest_lane_id]

        lane_start, lane_end, ego_dest_offset, lane_length = self.get_lane_start_end(ego_dest_lane_data,
                                                                                     ego_dest_offset)
        ego_dest_position = utils.compute_start_position(lane_start=lane_start, lane_end=lane_end,
                                                         lane_length=lane_length, offset=ego_dest_offset)
        x = ego_dest_position['x']
        y = 10.2
        z = ego_dest_position['z']
        # near_point_dest = self.sim.map_point_on_lane(lgsvl.Vector(x, y, z))
        self.destination = lgsvl.Vector(x, y,
                                        z)  # lgsvl.Vector(near_point_dest.position.x, near_point_dest.position.y, near_point_dest.position.z)

        # load mutated npc
        self.mutated_npc_list = []
        npc_size = lgsvl_input['npc_size']

        for npc_index in range(npc_size):
            lgsvl_lanes_to_follow = []
            lanes_to_follow = lgsvl_input['npc_lanes_to_follow'][npc_index]

            for l_index in range(len(lanes_to_follow)):
                lane_id = lanes_to_follow[l_index]
                lane_data = lgsvl_input['lanes'][lane_id]
                lane_start = self.convert_coordinate(point=lane_data['central']['points'][0])
                lane_end = self.convert_coordinate(point=lane_data['central']['points'][-1])
                lgsvl_lanes_to_follow.append({
                    "lane_id": lane_id,
                    "start": lgsvl.Vector(x=lane_start.position.x, y=10.2, z=lane_start.position.z),
                    "end": lgsvl.Vector(x=lane_end.position.x, y=10.2, z=lane_end.position.z),
                })

            # s1: add start point
            npc_type = lgsvl_input['npc_types'][npc_index]
            npc_waypoints = lgsvl_input['npc_waypoints'][npc_index]
            npc_start_lane_id = lanes_to_follow[0]
            npc_start_offset = npc_waypoints[0][0]
            npc_start_lane_data = lgsvl_input['lanes'][npc_start_lane_id]

            # logger.error('NPC ID: ' + str(npc_key) + 'Start Offset: ' + str(npc_start_offset))

            lane_start = self.convert_coordinate(point=npc_start_lane_data['central']['points'][0])
            lane_end = self.convert_coordinate(point=npc_start_lane_data['central']['points'][-1])
            lane_length = npc_start_lane_data['central']['length']
            npc_start_position = utils.compute_start_position(lane_start=lane_start, lane_end=lane_end,
                                                              lane_length=lane_length, offset=npc_start_offset)

            npc_pos_vector = lgsvl.Vector(x=npc_start_position['x'], y=10.2, z=npc_start_position['z'])
            npc_state = lgsvl.AgentState()
            npc_state.transform = self.sim.map_point_on_lane(npc_pos_vector)
            # logger.error('[Simulator] npc_type: ' + str(npc_type))
            npc = self.sim.add_agent(npc_type, lgsvl.AgentType.NPC, npc_state)

            wp = [{
                "lane_id": npc_start_lane_id,
                "start": lgsvl.Vector(x=lane_start.position.x, y=10.2, z=lane_start.position.z),
                "end": lgsvl.Vector(x=lane_end.position.x, y=10.2, z=lane_end.position.z),
                "offset": 0.0,
                "speed": npc_waypoints[0][1]
            }]
            # config waypoints
            # add wp -0 start point

            for m_index in range(1, len(npc_waypoints)):
                wp_offset = npc_waypoints[m_index][0]
                wp_speed = npc_waypoints[m_index][1]
                wp_lane_id = None
                lane_length = 0

                # find lane_id
                for l_index in range(len(lgsvl_input['npc_lanes_to_follow'][npc_index])):
                    lane_id = lgsvl_input['npc_lanes_to_follow'][npc_index][l_index]
                    lane_data = lgsvl_input['lanes'][lane_id]
                    lane_length += lane_data['central']['length']
                    # print(wp_offset, lane_length, lgsvl_input['npc_lanes_to_follow'][npc_index])
                    if wp_offset <= lane_length:
                        wp_lane_id = lane_id
                        break

                lane_data = lgsvl_input['lanes'][wp_lane_id]
                lane_start = self.convert_coordinate(point=lane_data['central']['points'][0])
                lane_end = self.convert_coordinate(point=lane_data['central']['points'][-1])

                adj_wp_offset = wp_offset - npc_start_offset

                wp_info = {
                    "lane_id": wp_lane_id,
                    "start": lgsvl.Vector(x=lane_start.position.x, y=10.2, z=lane_start.position.z),
                    "end": lgsvl.Vector(x=lane_end.position.x, y=10.2, z=lane_end.position.z),
                    "offset": adj_wp_offset,
                    "speed": wp_speed
                }

                wp.append(wp_info)

            self.mutated_npc_list.append({
                'npc_agent': npc,
                'waypoints': wp,
                'lanes_to_follow': lgsvl_lanes_to_follow
            })

        # load environments
        self.sim.weather = lgsvl.WeatherState(
            rain=lgsvl_input['environment']['rain'],
            fog=lgsvl_input['environment']['fog'],
            wetness=lgsvl_input['environment']['wetness'],
            cloudiness=lgsvl_input['environment']['cloudiness'],
            damage=lgsvl_input['environment']['damage']
        )
        self.sim.set_time_of_day(lgsvl_input['environment']['time'])

        # controllers
        controllables = self.sim.get_controllables()

        for i in range(len(controllables)):
            signal = controllables[i]
            if signal.type == "signal":
                # control_policy = signal.control_policy
                control_policy = "green=20;yellow=0;red=0;loop"  # fixed control policy
                signal.control(control_policy)

        self.yellow_lines = self.convert_legal_lines(lgsvl_input['yellow_lines'])
        self.cross_lines = self.convert_legal_lines(lgsvl_input['cross_lines'])
        self.edge_lines = self.convert_legal_lines(lgsvl_input['edge_lines'])
        logger.info('Load environment - Finish')

    def run(self, scenario_obj, case_id, target_recording_folder):

        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        logger.info('[Simulator] === Simulation Start:  [' + date_time + '] ===')

        self.simulation_count += 1
        self.load_map(self.lgsvl_map)
        # time.sleep(1)

        self.init_environment(scenario_obj)

        global collision_info
        global collision_flag

        def on_collision(agent1, agent2, contact):
            # TODO: add speed limitation
            global collision_info
            global collision_flag

            collision_info = {}

            name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
            name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
            logger.error('[Simulator] ' + str(name1) + " collided with " + str(name2) + " at " + str(contact))

            agent1_info = [agent1.state, agent1.bounding_box]

            if not agent2:
                agent2_info = [None, None]
            else:
                agent2_info = [agent2.state, agent2.bounding_box]

            if contact:
                contact_loc = [contact.x, contact.y, contact.z]
                collision_info['contact'] = contact_loc
            else:
                collision_info['contact'] = None

            collision_info['ego'] = agent1_info
            collision_info['npc'] = agent2_info

            collision_flag = True

        # INIT apollo
        self.ego.connect_bridge(address='127.0.0.1', port=9090)  # address, port
        self.ego.on_collision(on_collision)

        times = 0
        success = False
        dv = None
        while times < 3:
            try:
                dv = lgsvl.dreamview.Connection(self.sim, self.ego, os.environ.get("BRIDGE_HOST", "127.0.0.1"))
                dv.set_hd_map(self.apollo_map)
                dv.set_vehicle('Lincoln2017MKZ_LGSVL')
                dv.set_setup_mode('Mkz Lgsvl')
                dv.setup_apollo(self.destination.x, self.destination.z, self.modules, default_timeout=30)
                success = True
                break
            except:
                logger.warning('[Simulator] Fail to spin up apollo, try again!')
                times += 1
        if not success:
            raise RuntimeError('Fail to spin up apollo')

        dv.set_destination(self.destination.x, self.destination.z)
        logger.info(
            '[Simulator] Set Apollo (EGO) destination: ' + str(self.destination.x) + ',' + str(self.destination.z))

        # Frequency of action change of NPCs
        # 10hz
        for item in self.mutated_npc_list:
            npc = item['npc_agent']
            waypoints = item['waypoints']
            lanes_to_follow = item['lanes_to_follow']
            npc.follow_waypoints_offset(waypoints, lanes_to_follow)

        # start recording
        # utils.stop_recorder()
        recorder_folder_path = os.path.join(target_recording_folder, case_id)
        if not os.path.exists(recorder_folder_path):
            os.makedirs(recorder_folder_path)
        utils.start_recorder(os.path.join(recorder_folder_path, 'recording'))

        frame_id = 0
        time_limit = 200
        stop_frame_count = 0
        same_frame_time_count = 0
        type_continue_count = 0
        last_frame_sim_time = None
        collision_info = None
        collision_flag = False
        case_recorder = CaseRecorder(case_id)
        case_recorder.set_destination(self.destination)

        mutated_npc_num = len(self.mutated_npc_list)
        ego_bbox = self.ego.bounding_box
        npc_bboxes = []
        for npc_id in range(mutated_npc_num):
            npc_bboxes.append(self.mutated_npc_list[npc_id]['npc_agent'].bounding_box)

        delay_t = 3.0
        time.sleep(delay_t)

        if self.async_flag == 'async':
            self.sim.run_custom(time_limit)
            while True:
                is_stop = self.sim.check_status()
                frame_sim_time = self.sim.current_time
                # logger.info('frame_sim_time: ' + str(frame_sim_time))
                if not isinstance(frame_sim_time, float):
                    logger.info('frame_sim_time: ' + str(frame_sim_time))
                    type_continue_count += 1
                    if type_continue_count > 20:
                        case_recorder.add_case_event(CaseFaultType.UNKNOWN)
                        break
                    continue

                if frame_sim_time >= time_limit:
                    # time out
                    case_recorder.add_case_event(CaseFaultType.TIMEOUT)
                    break

                if last_frame_sim_time is None:
                    last_frame_sim_time = frame_sim_time
                else:
                    if last_frame_sim_time == frame_sim_time:
                        last_frame_sim_time = frame_sim_time
                        logger.warning('[Simulator]: Sample frequently -> continue sample ')
                        logger.warning('    current frame time: ' + str(frame_sim_time))
                        logger.warning('    last frame time: ' + str(last_frame_sim_time))
                        same_frame_time_count += 1
                        if same_frame_time_count > 10:
                            case_recorder.add_case_event(CaseFaultType.UNKNOWN)
                            break
                        else:
                            continue
                    else:
                        last_frame_sim_time = frame_sim_time
                        same_frame_time_count = 0

                frame_ego_state = self.ego.state
                frame_npc_info = []
                for npc_id in range(mutated_npc_num):
                    frame_npc_info.append({
                        'npc_id': npc_id,
                        'npc_bbox': npc_bboxes[npc_id],
                        'npc_state': self.mutated_npc_list[npc_id]['npc_agent'].state
                    })

                frame_element = FrameElement(frame_id, frame_sim_time, self.destination, ego_bbox, frame_ego_state,
                                             frame_npc_info, self.yellow_lines, self.edge_lines, self.cross_lines)
                frame_event, simulation_end_flag = case_recorder.add_frame(frame_element, collision=collision_flag)

                if frame_event == FrameEventType.STOP:
                    stop_frame_count += 1
                else:
                    stop_frame_count = 0

                if stop_frame_count > 300:
                    simulation_end_flag = True
                    case_recorder.add_case_event(CaseFaultType.LONG_STOP)

                if simulation_end_flag or collision_flag or is_stop:
                    break

                frame_id += 1

                time.sleep(0.05)
        elif self.async_flag == 'sync':
            while True:
                frame_sim_time = self.sim.current_time
                if frame_sim_time >= time_limit:
                    # time out
                    case_recorder.add_case_event(CaseFaultType.TIMEOUT)
                    break

                frame_ego_state = self.ego.state
                frame_npc_info = []
                for npc_id in range(mutated_npc_num):
                    frame_npc_info.append({
                        'npc_id': npc_id,
                        'npc_bbox': npc_bboxes[npc_id],
                        'npc_state': self.mutated_npc_list[npc_id]['npc_agent'].state
                    })

                frame_element = FrameElement(frame_id, frame_sim_time, self.destination, ego_bbox, frame_ego_state,
                                             frame_npc_info, self.yellow_lines, self.edge_lines, self.cross_lines)
                frame_event, simulation_end_flag = case_recorder.add_frame(frame_element, collision=collision_flag)

                if frame_event == FrameEventType.STOP:
                    stop_frame_count += 1
                else:
                    stop_frame_count = 0

                if stop_frame_count > 300:
                    simulation_end_flag = True
                    case_recorder.add_case_event(CaseFaultType.LONG_STOP)

                if simulation_end_flag or collision_flag:
                    break

                frame_id += 1
                self.sim.run(0.1)
        else:
            raise KeyError('Unsupported run mode: ' + str(self.async_flag))

        self.sim.stop()
        logger.info('simulation finished, total frames: ' + str(frame_id))

        # add to logger
        # module_status_mark = True
        # while module_status_mark:
        #     module_status_mark = False
        module_status = dv.get_module_status()
        for module, status in module_status.items():
            if (not status) and (module in self.modules):
                logger.warning('[Simulator] Module is closed: ' + module + ' ==> maybe not affect')
                # dv.enable_module(module)
                # time.sleep(0.5)
                # module_status_mark = True
                # case_recorder.add_case_event(CaseFaultType.MODULE_CLOSE)

        # time.sleep(0.5)
        utils.close_modules(dv, ['Recorder'])
        # utils.stop_recorder()

        if self.simulation_count % 50 == 0:
            logger.info('[Simulator] Restart all simulator modules in case high delays.')
            utils.close_modules(dv, self.modules)

        # offline analysis
        case_recorder.offline_analyze()
        logger.info('[Simulator] === Simulation End === ')

        return case_recorder

    def close(self):
        self.sim.close()
        self.sim = None