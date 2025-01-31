#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
from utils.config import color_dict, VISUAL_PATH
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.object_utils import get_nearby_moving_obj_feature_ls
from utils.lane_utils import get_nearby_lane_feature_ls, get_halluc_lane
from utils.viz_utils import show_doubled_lane, show_traj
from utils.agent_utils import get_agent_feature_ls
from utils.viz_utils import *
from utils.common import *
import pdb
import collections
from utils.config import *


def compute_feature_for_one_seq(name:str, traj_df: pd.DataFrame, am: ArgoverseMap,
                                obs_len: int = 20,
                                lane_radius: int = 5,
                                obj_radius: int = 10,
                                viz: bool = False,
                                mode='rect',
                                query_bbox=[-100, 100, -100, 100],
                                add_others: bool = False,
                                add_av: bool = False) -> List[List[List]]:
    """
    return lane & track features
    args:
        mode: 'rect' or 'nearby'
    returns:
        agent_feature_ls:
            list of (doubeld_track, object_type, timetamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
        norm_center np.ndarray: (3, )
    """
    res = []
    agent_track_id = None
    # normalize timestamps
    traj_df['TIMESTAMP'] -= np.min(traj_df['TIMESTAMP'].values)
    seq_ts = np.unique(traj_df['TIMESTAMP'].values)

    seq_len = seq_ts.shape[0]
    city_name = traj_df['CITY_NAME'].iloc[0]
    agent_df = None
    agent_x_end, agent_y_end, start_x, start_y, query_x, query_y, norm_center = [
        None] * 7
    others = collections.defaultdict(dict)
    # agent traj & its start/end point
    # for obj_type, remain_df in traj_df.groupby('OBJECT_TYPE'):
    
    for track_id, remain_df in traj_df.groupby('TRACK_ID'):
        obj_type = remain_df['OBJECT_TYPE'].iloc[0]
        if obj_type == 'AGENT':
            agent_track_id = track_id
            agent_df = remain_df
            # start_x, start_y = agent_df[['X', 'Y']].values[0]
            # agent_x_end, agent_y_end = agent_df[['X', 'Y']].values[-1]
            
            query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
            query_x_last, query_y_last = agent_df[['X', 'Y']].values[obs_len-2]
            
            vel_heading = np.arctan2(query_y - query_y_last, query_x - query_x_last)
            norm_center = np.array([query_x, query_y, vel_heading])
            # shift and rotate
            start_x, start_y = shift_and_rotate(agent_df[['X', 'Y']].values[0], norm_center[:2], norm_center[2])
            
            agent_x_end, agent_y_end = shift_and_rotate(agent_df[['X', 'Y']].values[-1], norm_center[:2], norm_center[2])
            
            break
        elif obj_type == 'OTHERS':
            if not add_others:
                continue
            if remain_df.shape[0] == 50:
                other_query_x, other_query_y = remain_df[['X', 'Y']].values[obs_len-1]
                other_query_x_last, other_query_y_last = remain_df[['X', 'Y']].values[obs_len-2]
                
                velocity = np.hypot(other_query_x - other_query_x_last, other_query_y - other_query_y_last) / (seq_ts[obs_len-1] - seq_ts[obs_len-2])
                if velocity < VELOCITY_THRESHOLD:
                    continue
                
                others[track_id]['df'] = remain_df
                vel_heading = np.arctan2(other_query_y - other_query_y_last, other_query_x - other_query_x_last)
                other_norm_center = np.array([other_query_x, other_query_y, vel_heading])
                others[track_id]['norm_center'] = other_norm_center

            # raise ValueError(f"cannot find 'agent' object type")

    # prune points after "obs_len" timestamp
    # [FIXED] test set length is only `obs_len`
    traj_df = traj_df[traj_df['TIMESTAMP'] <=
                      agent_df['TIMESTAMP'].values[obs_len-1]]

    assert (np.unique(traj_df["TIMESTAMP"].values).shape[0]
            == obs_len), "Obs len mismatch"

    # search nearby lane from the last observed point of agent
    # FIXME: nearby or rect?
    # lane_feature_ls = get_nearby_lane_feature_ls(
    #     am, agent_df, obs_len, city_name, lane_radius, norm_center)
    
    assert agent_track_id is not None
    lane_feature_ls = get_nearby_lane_feature_ls(
        am, agent_df, obs_len, city_name, lane_radius, norm_center, mode=mode, query_bbox=query_bbox)
    # pdb.set_trace()

    # search nearby moving objects from the last observed point of agent
    obj_feature_ls = get_nearby_moving_obj_feature_ls(
        agent_df, traj_df, obs_len, seq_ts, norm_center, agent_track_id)
    # get agent features
    agent_feature = get_agent_feature_ls(agent_df, obs_len, norm_center)
    res.append([agent_feature, obj_feature_ls, lane_feature_ls, norm_center, agent_track_id])
    
    if add_others:
        for track_id, track_info in others.items():
            other_lane_feature_ls = get_nearby_lane_feature_ls(
                am, track_info['df'], obs_len, city_name, lane_radius, track_info['norm_center'], mode=mode, query_bbox=query_bbox)
            # search nearby moving objects from the last observed point of agent
            other_obj_feature_ls = get_nearby_moving_obj_feature_ls(
                track_info['df'], traj_df, obs_len, seq_ts, track_info['norm_center'], track_id)
            # get agent features
            other_feature = get_agent_feature_ls(track_info['df'], obs_len, track_info['norm_center'])
            res.append([other_feature, other_obj_feature_ls, other_lane_feature_ls, track_info['norm_center'], track_id])
    # vis
    if viz:
        plt.figure(figsize=(20,16))
        plt.grid()
        for features in lane_feature_ls:
            show_doubled_lane(
                np.vstack((features[0][:, :2], features[0][-1, 3:5])))
            show_doubled_lane(
                np.vstack((features[1][:, :2], features[1][-1, 3:5])))
        for features in obj_feature_ls:
            show_traj(
                np.vstack((features[0][:, :2], features[0][-1, 2:])), features[1])
        show_traj(np.vstack(
            (agent_feature[0][:, :2], agent_feature[0][-1, 2:])), agent_feature[1])

        plt.plot(agent_x_end, agent_y_end, 'o',
                 color=color_dict['AGENT'], markersize=7)
        plt.plot(0, 0, 'x', color='blue', markersize=4)
        plt.plot(start_x, start_y,
                 'x', color='blue', markersize=4)
        # plt.show()
        plt.savefig(os.path.join(VISUAL_PATH, f'{name}.png'))

    # return [agent_feature, obj_feature_ls, lane_feature_ls, norm_center]
    return res


def trans_gt_offset_format(gt):
    """
    >Our predicted trajectories are parameterized as per-stepcoordinate offsets, starting from the last observed location.We rotate the coordinate system based on the heading of the target vehicle at the last observed location.
    
    """
    assert gt.shape == (30, 2) or gt.shape == (0, 2), f"{gt.shape} is wrong"

    # for test, no gt, just return a (0, 2) ndarray
    if gt.shape == (0, 2):
        return gt

    offset_gt = np.vstack((gt[0], gt[1:] - gt[:-1]))
    # import pdb
    # pdb.set_trace()
    assert (offset_gt.cumsum(axis=0) -
            gt).sum() < 1e-6, f"{(offset_gt.cumsum(axis=0) -gt).sum()}"

    return offset_gt


def encoding_features(agent_feature, obj_feature_ls, lane_feature_ls):
    """
    args:
        agent_feature_ls:
            list of (doubeld_track, object_type, timestamp, track_id, not_doubled_groudtruth_feature_trajectory)
        obj_feature_ls:
            list of list of (doubled_track, object_type, timestamp, track_id)
        lane_feature_ls:
            list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    returns:
        pd.DataFrame of (
            polyline_features: vstack[
                (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id),
                (xs, ys, xe, ye, NULL, zs, ze, polyline_id)
                ]
            offset_gt: incremental offset from agent's last obseved point,
            traj_id2mask: Dict[int, int]
            lane_id2mask: Dict[int, int]
        )
        where obejct_type = {0 - others, 1 - agent}

    """
    polyline_id = 0
    traj_id2mask, lane_id2mask = {}, {}
    gt = agent_feature[-1]
    traj_nd, lane_nd = np.empty((0, 7)), np.empty((0, 7))

    # encoding agent feature
    pre_traj_len = traj_nd.shape[0]
    agent_len = agent_feature[0].shape[0]
    # print(agent_feature[0].shape, np.ones(
    # (agent_len, 1)).shape, agent_feature[2].shape, (np.ones((agent_len, 1)) * polyline_id).shape)
    agent_nd = np.hstack((agent_feature[0], np.ones(
        (agent_len, 1)), agent_feature[2].reshape((-1, 1)), np.ones((agent_len, 1)) * polyline_id))
    assert agent_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"

    traj_nd = np.vstack((traj_nd, agent_nd))
    traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
    pre_traj_len = traj_nd.shape[0]
    polyline_id += 1

    # encoding obj feature
    for obj_feature in obj_feature_ls:
        obj_len = obj_feature[0].shape[0]
        # assert obj_feature[2].shape[0] == obj_len, f"obs_len of obj is {obj_len}"
        if not obj_feature[2].shape[0] == obj_len:
            from pdb import set_trace;set_trace()
        obj_nd = np.hstack((obj_feature[0], np.zeros(
            (obj_len, 1)), obj_feature[2].reshape((-1, 1)), np.ones((obj_len, 1)) * polyline_id))
        assert obj_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        traj_nd = np.vstack((traj_nd, obj_nd))

        traj_id2mask[polyline_id] = (pre_traj_len, traj_nd.shape[0])
        pre_traj_len = traj_nd.shape[0]
        polyline_id += 1

    # incodeing lane feature
    pre_lane_len = lane_nd.shape[0]
    for lane_feature in lane_feature_ls:
        l_lane_len = lane_feature[0].shape[0]
        l_lane_nd = np.hstack(
            (lane_feature[0], np.ones((l_lane_len, 1)) * polyline_id))
        assert l_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, l_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_1 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        r_lane_len = lane_feature[1].shape[0]
        r_lane_nd = np.hstack(
            (lane_feature[1], np.ones((r_lane_len, 1)) * polyline_id)
        )
        assert r_lane_nd.shape[1] == 7, "obj_traj feature dim 1 is not correct"
        lane_nd = np.vstack((lane_nd, r_lane_nd))
        lane_id2mask[polyline_id] = (pre_lane_len, lane_nd.shape[0])
        _tmp_len_2 = pre_lane_len - lane_nd.shape[0]
        pre_lane_len = lane_nd.shape[0]
        polyline_id += 1

        assert _tmp_len_1 == _tmp_len_2, f"left, right lane vector length contradict"
        # lane_nd = np.vstack((lane_nd, l_lane_nd, r_lane_nd))

    # FIXME: handling `nan` in lane_nd
    col_mean = np.nanmean(lane_nd, axis=0)
    if np.isnan(col_mean).any():
        # raise ValueError(
        # print(f"{col_mean}\nall z (height) coordinates are `nan`!!!!")
        lane_nd[:, 2].fill(.0)
        lane_nd[:, 5].fill(.0)
    else:
        inds = np.where(np.isnan(lane_nd))
        lane_nd[inds] = np.take(col_mean, inds[1])

    # traj_ls, lane_ls = reconstract_polyline(
    #     np.vstack((traj_nd, lane_nd)), traj_id2mask, lane_id2mask, traj_nd.shape[0])
    # type_ = 'AGENT'
    # for traj in traj_ls:
    #     show_traj(traj, type_)
    #     type_ = 'OTHERS'

    # for lane in lane_ls:
    #     show_doubled_lane(lane)
    # plt.show()

    # transform gt to offset_gt
    offset_gt = trans_gt_offset_format(gt)

    # now the features are:
    # (xs, ys, xe, ye, obejct_type, timestamp(avg_for_start_end?),polyline_id) for object
    # (xs, ys, zs, xe, ye, ze, polyline_id) for lanes

    # change lanes feature to xs, ys, xe, ye, NULL, zs, ze, polyline_id)
    lane_nd = np.hstack(
        [lane_nd, np.zeros((lane_nd.shape[0], 1), dtype=lane_nd.dtype)])
    lane_nd = lane_nd[:, [0, 1, 3, 4, 7, 2, 5, 6]]
    # change object features to (xs, ys, xe, ye, timestamp, NULL, NULL, polyline_id)
    traj_nd = np.hstack(
        [traj_nd, np.zeros((traj_nd.shape[0], 2), dtype=traj_nd.dtype)])
    traj_nd = traj_nd[:, [0, 1, 2, 3, 5, 7, 8, 6]]

    # don't ignore the id
    polyline_features = np.vstack((traj_nd, lane_nd))
    data = [[polyline_features.astype(
        np.float32), offset_gt, traj_id2mask, lane_id2mask, traj_nd.shape[0], lane_nd.shape[0]]]

    return pd.DataFrame(
        data,
        columns=["POLYLINE_FEATURES", "GT",
                 "TRAJ_ID_TO_MASK", "LANE_ID_TO_MASK", "TARJ_LEN", "LANE_LEN"]
    )


def save_features(df, name, dir_=None):
    if dir_ is None:
        dir_ = './input_data'
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    name = f"features_{name}.pkl"
    df.to_pickle(
        os.path.join(dir_, name)
    )
