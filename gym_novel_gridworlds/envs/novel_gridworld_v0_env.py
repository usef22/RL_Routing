import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import cv2

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from config import *
import time


def exit1():
    return None
    import sys
    sys.exit(0)


class NovelGridworldV0Env(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # NovelGridworldV0Env attributes
        self.env_name = 'NovelGridworld-v0'
        self.map_size = MAP_SIZE
        self.agent_location = (1, 1)  # row, column
        self.direction_id = {'NORTH': 0, 'SOUTH': 1, 'WEST': 2, 'EAST': 3}
        self.agent_facing = 0  # 'NORTH'
        self.agent_facing_str = list(self.direction_id.keys())[
            list(self.direction_id.values()).index(self.agent_facing)]
        self.block_in_front = 0  # Air
        self.map = np.zeros((self.map_size, self.map_size), dtype=int)  # 2D Map
        self.items_id = {'wall': 1, 'Destination': 2, "BW_Block": 3}  # ID cannot be 0 as air = 0
        self.items_quantity = {'Destination': 1,
                               "BW_Block": len(ids_set[0])}  # Do not include wall, quantity must be more than  0
        print("___________________________________________________________________________________________________")
        print(self.items_quantity)
        print("___________________________________________________________________________________________________")
        self.available_locations = []  # locations that do not have item placed
        self.not_available_locations = []  # locations that have item placed or are above, below, left, right to an item

        # Action Space
        # 0=Forward, 1=Left, 2=Right
        self.action_str = {0: 'Forward', 1: 'Left', 2: 'Right'}
        self.action_space = spaces.Discrete(len(self.action_str))
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken

        # Observation Space
        self.num_beams = 5
        self.max_beam_range = int(math.sqrt(2 * (self.map_size - 2) ** 2))  # Hypotonias of a square
        low = np.ones(len(self.items_id) * self.num_beams, dtype=int)
        high = np.array([self.max_beam_range] * len(self.items_id) * self.num_beams)
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.inventory_items_quantity = 0
        # Reward
        self.last_reward = 0  # last received reward

        self.last_done = False  # last done

        print(">>>>>>>>>>>>>>>>>>>>>>", self.map_size)
        self.traffic_map = traffic_map
        print(len(self.traffic_map))

        self.total_time_so_far = 0
        with open('Logs/time_sheet.txt', 'w+') as fd:
            fd.write(f'\n')

        exit1()
        #         self.traffic_map_img = self.get_traffic_map((self.map_size-2, self.map_size-2), self.traffic_map)
        print(self.traffic_map)

    #         cv2.imshow("Traffic map", self.traffic_map_img)
    #         cv2.waitKey(0)
    #         time.sleep(5)

    def map_traffic_delay(self, traffic):
        if traffic == 0:
            return 1.0
        else:
            #             previous_k = None
            for k in list(delay_index.keys()):
                if traffic >= k:
                    return delay_index[k]

    def get_traffic_map(self, grid_shape, trafic, color=(0, 255, 0), thickness=1):
        img = np.zeros((400, 400, 3))
        h, w, _ = img.shape
        rows, cols = grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        for a in range(rows + 1):
            for b in range(cols + 1):
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.7
                color = (255, 0, 0)
                thickness = 2
                image = cv2.putText(img, ("%d" % (trafic[a - 1, b - 1])), (
                int(a * dx) + int(img.shape[0] / rows * 0.4), int(b * dy) + int(img.shape[1] / cols * 0.6)), font,
                                    fontScale, color, thickness, cv2.LINE_AA)

        return img

    def reset(self, map_size=None, items_id=None, items_quantity=None, end=None, Agent=(None, None),
              table=(None, None)):

        if map_size is not None:
            self.map_size = map_size
        if items_id is not None:
            self.items_id = items_id
        if items_quantity is not None:
            self.items_quantity = items_quantity

        # Assertions and assumptions
        assert len(self.items_id) == len(self.items_quantity) + 1, "Should be equal, otherwise color might be wrong"

        # Variables to reset for each reset:
        self.available_locations = []
        self.not_available_locations = []
        self.last_action = 0  # last actions executed
        self.step_count = 0  # no. of steps taken
        self.last_reward = 0  # last received reward
        self.last_done = False  # last done

        self.map = np.zeros((self.map_size - 2, self.map_size - 2), dtype=int)  # air=0
        self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=self.items_id['wall'])

        # print(type(env.map), env.map.shape)
        # print(env.map)

        """
        available_locations: locations 1 block away from the wall are valid locations to place items and agent
        available_locations: locations that do not have item placed
        """
        for r in range(2, self.map_size - 2):
            for c in range(2, self.map_size - 2):
                self.available_locations.append((r, c))

        idx = 0
        self.agent_location = self.available_locations[idx]

        # print(self.agent_location, "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # Agent facing direction
        self.agent_facing = np.random.choice(4, size=1)[0]
        self.agent_facing_str = list(self.direction_id.keys())[
            list(self.direction_id.values()).index(self.agent_facing)]

        r, c = (None, None)

        for item, quantity in self.items_quantity.items():
            if item == 'Destination':
                idx_end = len(self.available_locations) - 1
                self.add_end_point_to_map(item, idx_end)
            else:
                a = np.random.choice(len(ids_set), size=1)[0]
                ids = ids_set[a]
                #                 print(a, ids)
                self.add_item_to_map(item, ids, r=r, c=c)

        if self.agent_location not in self.available_locations:
            self.available_locations.append(self.agent_location)

        observation = self.get_lidarSignal()
        self.inventory_items_quantity = 0

        self.total_time_so_far = 0

        return observation

    def add_end_point_to_map(self, item, idx):

        item_id = self.items_id[item]

        r, c = self.available_locations[idx]
        #         print(r, c)

        self.map[r][c] = item_id
        self.not_available_locations.append(self.available_locations.pop(idx))

    def add_item_to_map(self, item, ids, r, c):

        item_id = self.items_id[item]

        for r, c in ids:
            idx = self.available_locations.index((r, c))

            # If at (r, c) is air, and its North, South, West and East are also air, add item

            self.map[r][c] = item_id

            self.not_available_locations.append(self.available_locations.pop(idx))

    def add_item_to_map_(self, item, num_items, r=None, c=None):

        item_id = self.items_id[item]

        count = 0
        while True:
            if num_items == count:
                break
            #             print(len(self.available_locations), count)
            assert not len(self.available_locations) < 1, "Cannot place items, increase map size!"

            idx = np.random.choice(len(self.available_locations), size=1)[0]
            r, c = self.available_locations[idx]
            #                 print("at r")
            #                 print(r, c)
            #                 print(self.agent_location)

            if (r, c) == self.agent_location:
                r = None
                c = None
                self.available_locations.pop(idx)
                #                 print("at rc agent")
                continue

            # If at (r, c) is air, and its North, South, West and East are also air, add item
            if (self.map[r][c]) == 0 and (self.map[r - 1][c] == 0) and (self.map[r + 1][c] == 0) and (
                    self.map[r][c - 1] == 0) and (self.map[r][c + 1] == 0):
                self.map[r][c] = item_id
                count += 1
            self.not_available_locations.append(self.available_locations.pop(idx))

    def get_lidarSignal(self):
        """
        Send several beans (self.num_beams) at equally spaced angles in front of agent
        For each bean store distance (beam_range) for each item if item is found otherwise self.max_beam_range
        and return lidar_signals
        """
        direction_radian = {'NORTH': np.pi, 'SOUTH': 0, 'WEST': 3 * np.pi / 2, 'EAST': np.pi / 2}

        # All directions
        angles_list = np.linspace(direction_radian[self.agent_facing_str] - np.pi / 2,
                                  direction_radian[self.agent_facing_str] + np.pi / 2, self.num_beams)

        lidar_signals = []
        for angle in angles_list:
            x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)

            beam_range = 1
            # beam_signal = np.zeros(len(self.items_id), dtype=int)
            beam_signal = np.full(fill_value=self.max_beam_range, shape=len(self.items_id), dtype=int)

            # Keep sending longer beams until hit an object or wall
            while True:
                r, c = self.agent_location
                r_obj = r + np.round(beam_range * x_ratio)
                c_obj = c + np.round(beam_range * y_ratio)
                obj_id_rc = self.map[int(r_obj)][int(c_obj)]

                # If bean hit an object or wall
                if obj_id_rc != 0:
                    beam_signal[obj_id_rc - 1] = beam_range
                    break

                beam_range += 1

            lidar_signals.extend(beam_signal)

        return np.array(lidar_signals)

    def step(self, action):
        """
        Actions: 0=Forward, 1=Left, 2=Right
        """
        self.last_action = action
        r, c = self.agent_location
        #         print(r, c, "__________=====================___________________=====================_________")
        old_agent_location = copy.deepcopy(self.agent_location)
        old_agent_facing_str = copy.deepcopy(self.agent_facing_str)

        reward = -1  # If agent do not move/turn
        # Forward
        if action == 0:
            if self.agent_facing_str == 'NORTH' and self.map[r - 1][c] == 0:
                self.agent_location = [r - 1, c]
            elif self.agent_facing_str == 'SOUTH' and self.map[r + 1][c] == 0:
                self.agent_location = [r + 1, c]
            elif self.agent_facing_str == 'WEST' and self.map[r][c - 1] == 0:
                self.agent_location = [r, c - 1]
            elif self.agent_facing_str == 'EAST' and self.map[r][c + 1] == 0:
                self.agent_location = [r, c + 1]

            # if old_agent_location != self.agent_location:
            #     reward = -1  # If agent moved
        # Left
        elif action == 1:
            if self.agent_facing_str == 'NORTH':
                self.agent_facing_str = 'WEST'
            elif self.agent_facing_str == 'SOUTH':
                self.agent_facing_str = 'EAST'
            elif self.agent_facing_str == 'WEST':
                self.agent_facing_str = 'SOUTH'
            elif self.agent_facing_str == 'EAST':
                self.agent_facing_str = 'NORTH'

            # if old_agent_facing_str != self.agent_facing_str:
            #     reward = -1  # If agent turned
        # Right
        elif action == 2:
            if self.agent_facing_str == 'NORTH':
                self.agent_facing_str = 'EAST'
            elif self.agent_facing_str == 'SOUTH':
                self.agent_facing_str = 'WEST'
            elif self.agent_facing_str == 'WEST':
                self.agent_facing_str = 'NORTH'
            elif self.agent_facing_str == 'EAST':
                self.agent_facing_str = 'SOUTH'

        self.agent_facing = self.direction_id[self.agent_facing_str]
        self.find_block_in_front()

        block_r, block_c = self.get_block_in_front()
        observation = self.get_lidarSignal()

        if self.block_in_front == self.items_id['BW_Block']:
            delay_index_ = self.map_traffic_delay(self.traffic_map[r - 1, c - 1])
            self.inventory_items_quantity += (1000 * delay_index_) / self.traffic_map[r - 1, c - 1]

            if TEST_TYPE == "STEP":
                if self.inventory_items_quantity < BW_Req:
                    #                     reward = STEP_REWARD * (self.traffic_map[r, c] * delay_index_ ) / self.traffic_map[r, c]
                    reward = STEP_REWARD
            elif TEST_TYPE == "CUMULATIVE":
                if self.inventory_items_quantity >= BW_Req:
                    reward = CUMULATIVE_REWARD

            self.map[block_r][block_c] = 0

        reward -= int(self.traffic_map[r - 1, c - 1] // 600)
        done = False
        delay_index_ = self.map_traffic_delay(self.traffic_map[r - 1, c - 1])
        self.total_time_so_far += delay_index_

        if self.inventory_items_quantity >= BW_Req and self.block_in_front == self.items_id[
            'Destination']:
            reward = COMPLETION_REWARD
            with open('Logs/time_sheet.txt', 'a') as fd:
                fd.write(f'\n{str(self.total_time_so_far)}')
            self.total_time_so_far = 0
            done = True
        # elif self.block_in_front == self.items_id['tree']:
        #     reward = 0

        info = {}

        self.step_count += 1
        self.last_reward = reward
        self.last_done = done

        return observation, reward, done, info

    def find_block_in_front(self):
        r, c = self.agent_location

        if self.agent_facing_str == 'NORTH':
            self.block_in_front = self.map[r - 1][c]
        elif self.agent_facing_str == 'SOUTH':
            self.block_in_front = self.map[r + 1][c]
        elif self.agent_facing_str == 'WEST':
            self.block_in_front = self.map[r][c - 1]
        elif self.agent_facing_str == 'EAST':
            self.block_in_front = self.map[r][c + 1]

    def get_block_in_front(self):
        r, c = self.agent_location

        if self.agent_facing_str == 'NORTH':
            r1 = r - 1
            c1 = c
        elif self.agent_facing_str == 'SOUTH':
            r1 = r + 1
            c1 = c
        elif self.agent_facing_str == 'WEST':
            r1 = r
            c1 = c - 1
        elif self.agent_facing_str == 'EAST':
            r1 = r
            c1 = c + 1
        return r1, c1

    def render(self, mode='human'):
        r, c = self.agent_location
        print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- ", r, c)
        #         if r >= len(self.traffic_map):
        #             r = len(self.traffic_map)  - 1
        #         if c >= len(self.traffic_map):
        #             c = len(self.traffic_map)  - 1
        #         r -= 1
        #         c -= 1
        #         try:
        delay_index_ = self.map_traffic_delay(self.traffic_map[r - 1, c - 1])
        #         except:
        #             print(r, c)
        #             print(self.traffic_map)
        #             import sys
        #             time.sleep(1000)
        #             sys.exit(0)

        #         print(self.map_size, self.agent_location)
        traffic = (self.traffic_map[r - 1, c - 1] * delay_index_)
        print("trafic of *Sec : ", traffic, delay_index_)
        #         time.sleep((self.traffic_map[r, c] * delay_index_))
        self.map[r][c] = 0

        x2, y2 = 0, 0
        if self.agent_facing_str == 'NORTH':
            x2, y2 = 0, -0.01
        elif self.agent_facing_str == 'SOUTH':
            x2, y2 = 0, 0.01
        elif self.agent_facing_str == 'WEST':
            x2, y2 = -0.01, 0
        elif self.agent_facing_str == 'EAST':
            x2, y2 = 0.01, 0

        plt.figure(self.env_name, figsize=(9, 5))
        plt.imshow(self.map, cMAP="gist_ncar")
        plt.arrow(c, r, x2, y2, head_width=0.7, head_length=0.7, color='white')
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(self.map_size, self.map_size // 2, 'EAST', rotation=90)
        # plt.colorbar()
        # plt.grid()

        # legend_elements = [Line2D([0], [0], color='w', label="Agent Facing: " + self.agent_facing_str),
        #                    Line2D([0], [0], color='w', label="Action: " + self.action_str[self.action]),
        #                    Line2D([0], [0], color='w', label="Reward: " + str(self.reward))]
        # legend1 = plt.legend(handles=legend_elements, title="Info:", title_fontsize=12,
        #                      bbox_to_anchor=(1.62, 0.7))  # x, y

        info = '\n'.join(["               Info:             ",
                          "Steps: " + str(self.step_count),
                          "Agent Facing: " + self.agent_facing_str,
                          "Action: " + self.action_str[self.last_action],
                          "Reward: " + str(self.last_reward),
                          "Done: " + str(self.last_done),
                          "Traffic: " + str(traffic),
                          "Inventory: " + str(self.inventory_items_quantity)])
        props = dict(boxstyle='round', facecolor='w', alpha=0.2)
        plt.text(-(self.map_size // 2) - 0.5, 1.5, info, fontsize=10, bbox=props)

        cmap = get_cmap('gist_ncar')
        legend_elements = [Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=12,
                                  markeredgewidth=2, markeredgecolor='k')]
        for item in sorted(self.items_id):
            rgba = cmap(self.items_id[item] / len(self.items_id))
            legend_elements.append(
                Line2D([0], [0], marker="s", color='w', label=item, markerfacecolor=rgba, markersize=16))
        plt.legend(handles=legend_elements, title="Objects:", title_fontsize=12, bbox_to_anchor=(1.5, 1.02))  # x, y
        # plt.gca().add_artist(legend1)

        plt.tight_layout()
        #         plt.pause(0.001 * traffic)
        plt.pause(delay_index_)
        plt.clf()

    def close(self):
        return
