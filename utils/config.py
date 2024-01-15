#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}
LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
VELOCITY_THRESHOLD = 1.0
# Number of timesteps the track should exist to be considered in social context
EXIST_THRESHOLD = (50)
# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = (13)
color_dict = {"AGENT": "#d30000", "OTHERS": "#00d000", "AV": "#0000f2"}
LANE_RADIUS = 30
OBJ_RADIUS = 30
DATA_DIR = '../../data/argoverse1/vectornet'
OBS_LEN = 20
INTERMEDIATE_DATA_DIR = './interm_data'
# rotated by AGENT's vel_heaing
INTERMEDIATE_DATA_DIR = '../../data/argoverse1/vectornet/interm_data_rotated'

VISUAL_PATH = '../../data/argoverse1/vectornet/interm_data_rotated/visualize'

