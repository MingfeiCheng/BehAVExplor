import numpy as np
from loguru import logger
from shapely.geometry import LineString, Point

from common import utils

class FrameEventType:
    NORMAL = 0
    FINISH = 1
    COLLISION = 2
    DANGER_DISTANCE = 3
    YELLOW = 4
    EDGE = 5
    CROSS = 6
    LOW_SPEED = 7
    STOP = 8
    DEST_OVERTAKE = 9
    TIMEOUT = 10
    UNKNOWN = 11

class CaseFaultType:
    FINISH = 0
    COLLISION = 1
    DANGER_DISTANCE = 2
    HIT_YELLOW_LINE = 3
    HIT_EDGE_LINE = 4
    LONG_CROSS = 5
    DEST_OVERTAKE = 6
    DEST_UNREACH = 7
    FAST_ACC = 8
    HARD_BRAKE = 9
    MODULE_CLOSE = 10
    LOW_SPEED = 11
    LONG_STOP = 12
    OVER_SPEED = 13
    TIMEOUT = 14
    UNKNOWN = 15
    LATENCY = 16

class CaseRecorder(object):

    def __init__(self, case_id):
        self.case_id = case_id
        self.case_event = []
        self.frames = []

        self.destination = None
        self.last_frame_dist2dest = None

        self.min_distance2NPC = 99999

        # index
        self.collision_frame = None
        self.danger_frames = []
        self.hit_yellow_frames = []
        self.hit_edge_frames = []

        self.low_speed_frames = []
        self.max_low_speed_frame_num = 0

        self.case_vector = None # T, N

        self.case_fitness = [] # last_frame_dist2dest, min_npc_distance, yellow_line, edge_line, max_cross_rate, max_stop_rate, speed_max, acc_max, acc_min

    def get_case_id(self):
        return self.case_id

    def set_destination(self, destination):
        self.destination = Point(destination.x, destination.z)
        self.last_frame_dist2dest = None

    def add_case_event(self, fault):
        self.case_event.append(fault)

    def get_case_event(self):
        self.case_event = sorted(list(set(self.case_event)))
        return self.case_event

    def get_case_vector(self):
        return self.case_vector

    def get_case_fitness(self):
        return self.case_fitness

    def obtain_case_event_str(self):
        case_events = self.get_case_event()
        event_str = []
        for event in case_events:
            if event == CaseFaultType.FINISH:
                event_str.append('finish')
            elif event == CaseFaultType.COLLISION:
                event_str.append('collision')
            elif event == CaseFaultType.DANGER_DISTANCE:
                event_str.append('danger_distance')
            elif event == CaseFaultType.HIT_YELLOW_LINE:
                event_str.append('hit_yellow_line')
            elif event == CaseFaultType.HIT_EDGE_LINE:
                event_str.append('hit_edge_line')
            elif event == CaseFaultType.LONG_CROSS:
                event_str.append('long_cross')
            elif event == CaseFaultType.DEST_OVERTAKE:
                event_str.append('dest_overtake')
            elif event == CaseFaultType.DEST_UNREACH:
                event_str.append('dest_unreach')
            elif event == CaseFaultType.FAST_ACC:
                event_str.append('fast_acc')
            elif event == CaseFaultType.HARD_BRAKE:
                event_str.append('hard_brake')
            elif event == CaseFaultType.MODULE_CLOSE:
                event_str.append('module_close')
            elif event == CaseFaultType.LOW_SPEED:
                event_str.append('low_speed')
            elif event == CaseFaultType.LONG_STOP:
                event_str.append('long_stop')
            elif event == CaseFaultType.OVER_SPEED:
                event_str.append('over_speed')
            elif event == CaseFaultType.TIMEOUT:
                event_str.append('timeout')
            elif event == CaseFaultType.UNKNOWN:
                event_str.append('unknown')
            elif event == CaseFaultType.LATENCY:
                event_str.append('latency')

        return ' '.join(event_str)

    def add_frame(self, frame_element, collision=False):
        event_type = FrameEventType.NORMAL
        end_simulation = False

        current_dist = frame_element.get_distance2Destination()
        ego_speed = frame_element.get_ego_speed()

        if collision:
            frame_element.set_online_event(FrameEventType.COLLISION)
            self.frames.append(frame_element)

            event_type = FrameEventType.COLLISION
            end_simulation = True
            return event_type, end_simulation

        if self.last_frame_dist2dest is None:
            self.last_frame_dist2dest = current_dist
            event_type = FrameEventType.NORMAL
            end_simulation = False
        else:
            if current_dist <= 2 and ego_speed <= 0.05:
                event_type = FrameEventType.FINISH
                end_simulation = True

            elif (current_dist - self.last_frame_dist2dest) > 0.06:
                event_type = FrameEventType.DEST_OVERTAKE
                end_simulation = True

            elif -0.5 <= (current_dist - self.last_frame_dist2dest) <= 0.06 and current_dist > 2:
                if ego_speed < 0.3:
                    event_type = FrameEventType.STOP
                    end_simulation = False
                elif ego_speed < 1.0:
                    event_type = FrameEventType.LOW_SPEED
                    end_simulation = False

        frame_element.set_online_event(event_type)
        self.frames.append(frame_element)
        # logger.info('current_dist - self.last_frame_dist2dest: ' + str(current_dist - self.last_frame_dist2dest))
        # logger.info('event_type: ' + str(event_type))
        self.last_frame_dist2dest = current_dist
        return event_type, end_simulation

    def offline_analyze(self):
        total_frames = len(self.frames)

        case_min_npc_distance = 999999
        case_min_yellow_distance = 999999
        case_min_edge_distance = 999999
        case_max_speed = -999999
        case_max_acc = -999999
        case_min_acc = 999999

        self.case_vector = []
        last_frame = self.frames[-1]
        last_frame_dist2dest = last_frame.get_distance2Destination()
        if last_frame_dist2dest > 10.0:
            self.add_case_event(CaseFaultType.DEST_UNREACH)
        
        stop_array = [] # 0 not stop 1 stop
        cross_array = [] # 0 not cross 1 cross
        start_time = self.frames[0].get_sim_time()

        for i in range(total_frames):
            frame_i = self.frames[i]
            frame_i.reset_sim_time(start_time)
            frame_events = frame_i.compute_offline_event()

            current_frame_cross_event = False
            current_frame_stop_event = False

            for fe in frame_events:
                if fe == FrameEventType.COLLISION:
                    self.add_case_event(CaseFaultType.COLLISION)
                    self.collision_frame = i
                elif fe == FrameEventType.DANGER_DISTANCE:
                    self.add_case_event(CaseFaultType.DANGER_DISTANCE)
                    self.danger_frames.append(i)
                elif fe == FrameEventType.YELLOW:
                    self.add_case_event(CaseFaultType.HIT_YELLOW_LINE)
                    self.hit_yellow_frames.append(i)
                elif fe == FrameEventType.EDGE:
                    self.add_case_event(CaseFaultType.HIT_EDGE_LINE)
                    self.hit_edge_frames.append(i)
                elif fe == FrameEventType.CROSS:
                    current_frame_cross_event = True
                elif fe == FrameEventType.STOP:
                    current_frame_stop_event = True
                elif fe == FrameEventType.LOW_SPEED:
                    self.low_speed_frames.append(i)
                    self.max_low_speed_frame_num += 1
                elif fe == FrameEventType.DEST_OVERTAKE:
                    self.add_case_event(CaseFaultType.DEST_OVERTAKE)
                else:
                    pass
            
            if current_frame_cross_event:
                cross_array.append(1)
            else:
                cross_array.append(0)
            
            if current_frame_stop_event:
                stop_array.append(1)
            else:
                stop_array.append(0)

            if i == len(self.frames) - 1:
                frame_i.calculate_ego_attributes(next_frame=None)
            else:
                frame_i.calculate_ego_attributes(next_frame=self.frames[i + 1])

            frame_i_vector = frame_i.get_frame_vector()
            if self.case_vector is None or len(self.case_vector) <= 0:
                self.case_vector = np.array([frame_i_vector])
            else:
                self.case_vector = np.concatenate([self.case_vector, np.array([frame_i_vector])], axis=0)

            # fast acc & hard brake
            frame_i_acc = frame_i.get_ego_acceleration()
            if frame_i_acc >= 6.95:
                self.add_case_event(CaseFaultType.FAST_ACC)
            elif frame_i_acc <= -6.95:
                self.add_case_event(CaseFaultType.HARD_BRAKE)

            # obtain case_fitness
            frame_i_speed = frame_i.get_ego_speed()
            if frame_i_speed > case_max_speed:
                case_max_speed = frame_i_speed

            if frame_i_acc > case_max_acc:
                case_max_acc = frame_i_acc
            elif frame_i_acc < case_min_acc:
                case_min_acc = frame_i_acc

            frame_i_min_npc_distance = frame_i.get_min_distance2NPCs()
            if frame_i_min_npc_distance < case_min_npc_distance:
                case_min_npc_distance = frame_i_min_npc_distance

            frame_i_distance_yellow = frame_i.get_distance2Yellow()
            if frame_i_distance_yellow < case_min_yellow_distance:
                case_min_yellow_distance = frame_i_distance_yellow

            frame_i_distance_edge = frame_i.get_distance2Edge()
            if frame_i_distance_edge < case_min_edge_distance:
                case_min_edge_distance = frame_i_distance_edge

        # Analyse low speed, stop, cross line
        # low speed
        low_speed_frame_rates = len(self.low_speed_frames) / float(total_frames)
        if low_speed_frame_rates >= 0.6:
            self.add_case_event(CaseFaultType.LOW_SPEED)

        # stop
        window_size = 200
        max_stop_rate = 0
        for i in range(total_frames - window_size):
            stop_window = stop_array[i:i + window_size]
            stop_count = sum(stop_window)
            stop_rate = stop_count / float(window_size)
            if stop_rate > max_stop_rate:
                max_stop_rate = stop_rate
        if max_stop_rate > 0.95:
            self.add_case_event(CaseFaultType.LONG_STOP)
        
        window_size = 70
        max_cross_rate = 0
        for i in range(total_frames - window_size):
            cross_window = cross_array[i:i + window_size]
            cross_count = sum(cross_window)
            cross_rate = cross_count / float(window_size)
            if cross_rate > max_cross_rate:
                max_cross_rate = cross_rate
        if max_cross_rate > 0.95:
            self.add_case_event(CaseFaultType.LONG_CROSS)
        
        if case_max_speed > 12:
            self.add_case_event(CaseFaultType.OVER_SPEED)

        if len(self.case_event) == 0:
            logger.debug('Case event is Normal')
            self.add_case_event(CaseFaultType.FINISH)

        # last_frame_dist2dest, min_npc_distance, yellow_line, edge_line, max_cross_rate, max_stop_rate, speed_max, acc_max, acc_min
        self.case_fitness = np.array([last_frame_dist2dest, case_min_npc_distance, case_min_yellow_distance, case_min_edge_distance, max_cross_rate, max_stop_rate, case_max_speed, case_max_acc, case_min_acc])
        self.case_vector = np.array(self.case_vector)

    def get_time_scale(self):
        start_frame_time = self.frames[0].get_sim_time()
        end_frame_time = self.frames[-1].get_sim_time()
        assert end_frame_time > start_frame_time
        return [start_frame_time, end_frame_time]

class FrameElement(object):

    def __init__(self, frame_id, sim_time, destination, ego_bbox, ego_state, npc_info, yellow_lines, edge_lines, cross_lines):
        self.frame_id = frame_id
        self.logical_time = frame_id * 0.1
        self.sim_time = sim_time
        self.ego_bbox = ego_bbox
        self.ego_state = ego_state
        self.npc_info = npc_info
        self.yellow_lines = yellow_lines
        self.edge_lines = edge_lines
        self.cross_lines = cross_lines

        self.online_event = None
        self.offline_event = []

        self.destination = Point(destination.x, destination.z)
        self.distance2NPCs = []
        self.min_distance2NPCs = 999999
        self.distance2Yellow = 999999
        self.distance2Edge = 999999
        self.distance2Cross = 999999
        self.distance2Destination = 999999

        self.ego_speed = None
        self.ego_size = None
        self.ego_velocity = None
        self.ego_velocity_hl = None
        self.ego_angular_velocity = None
        self.ego_heading = None
        self.ego_acceleration = None
        self.ego_acceleration_vector = None
        self.ego_acceleration_vector_hl = None
        self.ego_position = None

        self.npc_attributions = None

    def set_online_event(self, event):
        self.online_event = event
        self.offline_event.append(event)

    def compute_offline_event(self):
        # compute NPC distance
        adjust_ego_bbox = utils.get_bbox(self.ego_state, self.ego_bbox)

        if self.online_event == FrameEventType.COLLISION:
            self.min_distance2NPCs = 0.0
            self.offline_event.append(FrameEventType.COLLISION)
        else:
            for i in range(len(self.npc_info)):
                adjust_npc_bbox = utils.get_bbox(self.npc_info[i]['npc_state'], self.npc_info[i]['npc_bbox'])
                distance2npc_i = adjust_ego_bbox.distance(adjust_npc_bbox)
                if distance2npc_i < self.min_distance2NPCs:
                    self.min_distance2NPCs = distance2npc_i
                self.distance2NPCs.append(distance2npc_i)

            if 0 < self.min_distance2NPCs < 0.5:
                self.offline_event.append(FrameEventType.DANGER_DISTANCE)

        # compute yellow line distance
        for yellow_line in self.yellow_lines:
            adjust_yellow_line = LineString(yellow_line)
            distance2yellow = adjust_ego_bbox.distance(adjust_yellow_line)
            if distance2yellow < self.distance2Yellow:
                self.distance2Yellow = distance2yellow
        if self.distance2Yellow <= 0:
            self.offline_event.append(FrameEventType.YELLOW)

        # compute edge line distance
        for edge_line in self.edge_lines:
            adjust_edge_line = LineString(edge_line)
            distance2edge = adjust_ego_bbox.distance(adjust_edge_line)
            if distance2edge < self.distance2Edge:
                self.distance2Edge = distance2edge
        if self.distance2Edge <= 0:
            self.offline_event.append(FrameEventType.EDGE)

        for cross_line in self.cross_lines:
            adjust_cross_line = LineString(cross_line)
            distance2cross = adjust_ego_bbox.distance(adjust_cross_line)
            if distance2cross < self.distance2Cross:
                self.distance2Cross = distance2cross

        if self.distance2Cross <= 0:
            self.offline_event.append(FrameEventType.CROSS)

        # judge destination
        if self.distance2Destination == 999999 or self.ego_speed is None:
            logger.warning('Frame to destination : ' + str(self.distance2Destination))
            logger.warning('Frame Ego speed : ' + str(self.ego_speed))
            _ = self.get_distance2Destination()
            _ = self.get_ego_speed()
            logger.warning('Re-Calculating...')
            logger.warning('Frame to destination : ' + str(self.distance2Destination))
            logger.warning('Frame Ego speed : ' + str(self.ego_speed))

        if self.distance2Destination > 3:
            if self.ego_speed <= 0.3:
                self.offline_event.append(FrameEventType.STOP)
            elif 0.3 <= self.ego_speed < 1.0:
                self.offline_event.append(FrameEventType.LOW_SPEED)

        self.offline_event.append(self.online_event)
        return list(set(self.offline_event))

    def get_distance2NPCs(self):
        return self.distance2NPCs

    def get_min_distance2NPCs(self):
        return self.min_distance2NPCs

    def get_distance2Yellow(self):
        return self.distance2Yellow

    def get_distance2Edge(self):
        return self.distance2Edge

    def get_distance2Cross(self):
        return self.distance2Cross

    def get_distance2Destination(self):
        if self.distance2Destination == 999999:
            adjust_ego_bbox = utils.get_bbox(self.ego_state, self.ego_bbox)
            self.distance2Destination = adjust_ego_bbox.distance(self.destination)

        return self.distance2Destination

    def get_ego_speed(self):
        if self.ego_speed is None:
            self.ego_speed = np.linalg.norm(np.array([self.ego_state.velocity.x, self.ego_state.velocity.z])) #np.linalg.norm(np.array([self.ego_state.velocity.x, self.ego_state.velocity.y, self.ego_state.velocity.z]))
        return self.ego_speed

    def get_ego_acceleration(self):
        return self.ego_acceleration

    def get_sim_time(self):
        return self.sim_time

    def get_ego_velocity(self):
        if self.ego_velocity is None:
            self.ego_velocity = np.array([self.ego_state.velocity.x, self.ego_state.velocity.z]) # np.array([self.ego_state.velocity.x, self.ego_state.velocity.y, self.ego_state.velocity.z])
        return self.ego_velocity

    def get_ego_velocity_hl(self):
        if self.ego_velocity_hl is None:
            self.ego_velocity_hl = utils.ego_speed_converter(self.ego_state)  # np.array([self.ego_state.velocity.x, self.ego_state.velocity.y, self.ego_state.velocity.z])
        return self.ego_velocity_hl

    def get_ego_angular_velocity(self):
        if self.ego_angular_velocity is None:
            self.ego_angular_velocity = np.array([self.ego_state.angular_velocity.x, self.ego_state.angular_velocity.y, self.ego_state.angular_velocity.z])
        return self.ego_angular_velocity

    def get_ego_heading(self):
        if self.ego_heading is None:
            self.ego_heading = np.array([self.ego_state.transform.rotation.x, self.ego_state.transform.rotation.y, self.ego_state.transform.rotation.z])
        return self.ego_heading

    def get_ego_position(self):
        if self.ego_position is None:
            self.ego_position = np.array([self.ego_state.transform.position.x, self.ego_state.transform.position.y,
                                          self.ego_state.transform.position.z])
        return self.ego_position

    def get_ego_size(self):
        if self.ego_size is None:
            ego_height = self.ego_bbox.max.y - self.ego_bbox.min.y
            ego_width = self.ego_bbox.max.z - self.ego_bbox.min.z
            ego_length = self.ego_bbox.max.x - self.ego_bbox.min.x
            self.ego_size = np.array([ego_length, ego_width, ego_height])
        return self.ego_size

    def calculate_ego_attributes(self, next_frame=None):
        ego_speed = self.get_ego_speed()
        ego_velocity = self.get_ego_velocity()
        ego_angular_velocity = self.get_ego_angular_velocity()
        ego_heading = self.get_ego_heading()
        ego_position = self.get_ego_position()
        ego_size = self.get_ego_size()
        ego_velocity_hl = self.get_ego_velocity_hl()

        if next_frame is not None:
            next_frame_ego_velocity = next_frame.get_ego_velocity()
            next_frame_ego_speed = next_frame.get_ego_speed()
            next_frame_sim_time = next_frame.get_sim_time()
        else:
            # optimize, None means next frame, ideal is 0 for all
            next_frame_sim_time = self.sim_time + 0.1
            next_frame_ego_speed = 0.0
            next_frame_ego_velocity = np.array([0.0, 0.0])

        if next_frame_sim_time == self.sim_time:
            logger.warning('[FrameElement] sim times have conflict! self.sim_time: ' + str(
                self.sim_time) + ' next_frame_sim_time: ' + str(next_frame_sim_time))

        self.ego_acceleration_vector = (next_frame_ego_velocity - ego_velocity) / (next_frame_sim_time - self.sim_time + 1e-10)
        self.ego_acceleration = np.linalg.norm(self.ego_acceleration_vector) * (-1 if next_frame_ego_speed < ego_speed else 1)
        self.ego_acceleration_vector_hl = utils.ego_accelerate_converter(self.ego_state, self.ego_acceleration_vector)  # np.array ([head, lateral])

    def get_frame_vector(self):
        vector = np.array([self.sim_time, self.ego_speed, self.ego_acceleration]) # [0, 1, 2]
        vector = np.concatenate([vector, self.ego_position]) # [3, 4, 5]
        vector = np.concatenate([vector, self.ego_velocity_hl]) # [6, 7]
        vector = np.concatenate([vector, self.ego_acceleration_vector_hl]) # [8, 9]
        vector = np.concatenate([vector, self.ego_angular_velocity]) # [10, 11, 12]
        vector = np.concatenate([vector, self.ego_heading]) # [13, 14, 15]
        return vector

    def reset_sim_time(self, start_time):
        self.sim_time = self.sim_time - start_time

    def get_frame_env_vector(self):
        pass

    def calculate_npc_attributes(self):
        pass