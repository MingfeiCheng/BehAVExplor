import os
import time
import math
import numpy as np

from shapely.geometry import Polygon
from loguru import logger

def close_modules(dv, modules):
    not_all = True
    while not_all:
        not_all = False
        module_status = dv.get_module_status()
        for module, status in module_status.items():
            if status and (module in modules):
                dv.disable_module(module)
                not_all = True
        time.sleep(0.5)

def start_recorder(output_path):

    os.system('nohup cyber_recorder record -a -o ' + output_path + ' &')
    os.system('ps -ef|grep cyber_recorder|grep -v grep > check_recorder.txt')

    file_size = int(os.path.getsize('check_recorder.txt'))
    while file_size == 0:
        os.system('nohup cyber_recorder record -a -o ' + output_path + ' &')
        os.system('ps -ef|grep cyber_recorder|grep -v grep > check_recorder.txt')
        file_size = int(os.path.getsize('check_recorder.txt'))

def stop_recorder():
    os.system("kill $(ps -ef|grep cyber_recorder | grep -v grep | awk {'print$2'})")

def calc_abc_from_line_2d(x0, y0, x1, y1):
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c

def get_line_cross_point(line1, line2):
    a0, b0, c0 = calc_abc_from_line_2d(*line1)
    a1, b1, c1 = calc_abc_from_line_2d(*line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y

def right_rotation(coord, theta):
    """
    theta : degree
    """
    theta = math.radians(theta)
    x = coord[1]
    y = coord[0]
    x1 = x * math.cos(theta) - y * math.sin(theta)
    y1 = x * math.sin(theta) + y * math.cos(theta)
    return [y1, x1]

def ego_speed_converter(agent_state):
    """
    return np.array ([head, lateral])
    """
    agent_theta = agent_state.transform.rotation.y
    velocity_x = np.array([agent_state.velocity.x, 0])
    velocity_z = np.array([0, agent_state.velocity.z])

    head_vector = [1, 0]
    lateral_vector = [0, 1]

    head_vector = np.array(right_rotation(head_vector, agent_theta))
    lateral_vector = np.array(right_rotation(lateral_vector, agent_theta))

    velocity_head = np.dot(head_vector, velocity_x) + np.dot(head_vector, velocity_z)
    velocity_lateral = np.dot(lateral_vector, velocity_x) + np.dot(lateral_vector, velocity_z)

    return np.array([velocity_head, velocity_lateral])

def ego_accelerate_converter(agent_state, acceleration):
    """
    acceleration: [x, z]
    return np.array ([head, lateral])
    """
    agent_theta = agent_state.transform.rotation.y
    acceleration_x = np.array([acceleration[0], 0])
    acceleration_z = np.array([0, acceleration[1]])

    head_vector = [1, 0]
    lateral_vector = [0, 1]

    head_vector = np.array(right_rotation(head_vector, agent_theta))
    lateral_vector = np.array(right_rotation(lateral_vector, agent_theta))

    acceleration_head = np.dot(head_vector, acceleration_x) + np.dot(head_vector, acceleration_z)
    acceleration_lateral = np.dot(lateral_vector, acceleration_x) + np.dot(lateral_vector, acceleration_z)

    return np.array([acceleration_head, acceleration_lateral])

def get_bbox(agent_state, agent_bbox):
    agent_theta = agent_state.transform.rotation.y
    #agent_bbox min max (x_min, y_min, z_min) (x_max, y_max, z_max)

    global_x = agent_state.transform.position.x
    global_z = agent_state.transform.position.z
    x_min = agent_bbox.min.x + 0.1
    x_max = agent_bbox.max.x - 0.1
    z_min = agent_bbox.min.z + 0.1
    z_max = agent_bbox.max.z - 0.1

    line1 = [x_min, z_min, x_max, z_max]
    line2 = [x_min, z_max, x_max, z_min]
    x_center, z_center = get_line_cross_point(line1, line2)

    coords = [[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]]
    new_coords = []

    for i in range(len(coords)):
        coord_i = coords[i]
        coord_i[0] = coord_i[0] - x_center
        coord_i[1] = coord_i[1] - z_center
        new_coord_i = right_rotation(coord_i, agent_theta)
        new_coord_i[0] += global_x
        new_coord_i[1] += global_z
        new_coords.append(new_coord_i)
    p1, p2, p3, p4 = new_coords[0], new_coords[1], new_coords[2], new_coords[3]

    agent_poly = Polygon((p1, p2, p3, p4))
    if agent_poly.area <= 0:
        logger.warning('agent_poly area <= 0')
    return agent_poly


def compute_start_position(lane_start, lane_end, lane_length, offset):
    """
        lane_start: transform obj
    """
    if offset == 0:
        x = lane_start.position.x
        z = lane_start.position.z
    else:
        v_x = lane_end.position.x - lane_start.position.x
        v_z = lane_end.position.z - lane_start.position.z
        ratio = offset / (lane_length + 0.0)
        x = lane_start.position.x + ratio * v_x
        z = lane_start.position.z + ratio * v_z

    return {'x': x, 'z': z}

def interval_intersection(A, B):
    ans = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Let's check if A[i] intersects B[j].
        # lo - the startpoint of the intersection
        # hi - the endpoint of the intersection
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            ans.append([lo, hi])

        # Remove the interval with the smallest endpoint
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return ans

def check_relocate(offset_recorder, offset, lane_id, offset_min, offset_max):

    if not lane_id in offset_recorder.keys():
        lane_interval = [[offset_min, offset_max]]
        offset_recorder[lane_id] = []
        offset_recorder[lane_id].append(lane_interval)
        offset_recorder[lane_id].append([[-200, offset - 8], [offset + 8, 9999]])
        relocated_offset = float(np.clip(offset, offset_min, offset_max))

        return relocated_offset, offset_recorder
    else:

        lane_offsets = offset_recorder[lane_id]
        available_interval = []
        for i in range(1, len(lane_offsets)):
            if i == 1:
                available_interval = interval_intersection(lane_offsets[i], lane_offsets[i - 1])
            else:
                available_interval = interval_intersection(available_interval, lane_offsets[i])

        if len(available_interval) <= 0:
            logger.warning(
                '[INPUT CORPUS] Please check the map and location of the NPCs. The scenario config is not suitable.')

        min_dist = 99999
        interval_index = -1
        for i in range(len(available_interval)):
            # compute nearest interval
            dist = min([abs(offset - available_interval[i][0]), abs(offset - available_interval[i][1])])
            if dist < min_dist:
                min_dist = dist
                interval_index = i
        relocate_interval = available_interval[interval_index]
        relocated_offset = float(np.clip(offset, relocate_interval[0], relocate_interval[1]))
        offset_recorder[lane_id].append([[-200, relocated_offset - 8], [relocated_offset + 8, 9999]])

        return relocated_offset, offset_recorder

