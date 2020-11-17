""" Create a train set. """

import random
import numpy as np
from animal.arenas.utils import (

    # random
    create_c1_arena,
    create_c1_arena_weird,
    create_c2_arena,
    create_c3_arena,
    create_c3_arena_basic,
    create_c4_arena,
    create_c5_arena,
    create_c6_arena,
    create_c6_arena_basic,
    create_c7_arena,

    # skills
    create_maze,
    create_arena_choice,
    create_arena_cross,
    create_arena_push1,
    create_arena_push2,
    create_arena_tunnel1,
    create_arena_tunnel2,
    create_arena_ramp1,
    create_arena_ramp2,
    create_arena_ramp3,
    create_arena_narrow_spaces_1,
    create_arena_narrow_spaces_2,
    create_arena_pref1,
    create_blackout_test_1,
    create_reasoning_step_1,
    create_reasoning_step_2,
    create_reasoning_step_3,

    # more skills
    create_left_right,
    create_front_back,
    create_corners_green,
    create_cross_green,
    create_in_front,
    create_make_fall_1,
    create_arena_choice_2,
)

if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas train directory')
    parser.add_argument(
        '-u', '--unify', action='store_true', default=False,
        help='Save all arenas in the same directory')
    parser.add_argument(
        '-o', '--only-specific', action='store_true', default=False,
        help='Create only arenas of specific skills')
    parser.add_argument(
        '-p', '--phases', action='store_true', default=False,
        help='Create arenas to train skills in phases')
    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)

    if arguments.unify:
        save_in = arguments.target_dir
    else:

        if arguments.only_specific:
            skills = ["push", "ramps", "narrow", "tunnels", "reasoning",
                      "navigate", "mazes", "blackouts", "make_fall"]
        else:
            skills = ["preferences", "push", "ramps", "narrow", "zones",
                      "tunnels", "navigate", "generalize", "internal_model",
                      "mazes", "choices", "reasoning", "blackouts", "make_fall"]

        if arguments.phases:
            skills += ["train_in_phases"]

        for skill in skills:
            if not os.path.isdir("{}/{}".format(arguments.target_dir, skill)):
                os.mkdir("{}/{}".format(arguments.target_dir, skill))

    print("Creating preference arenas...")

    if not arguments.only_specific:
        # c1
        for i in range(1, 201):
            create_c1_arena(
                arguments.target_dir if arguments.unify
                else "{}/preferences/".format(arguments.target_dir),
                'c1_{}'.format(str(i).zfill(4)),
                max_reward=float(np.random.randint(5, 10)),
                time=random.choice([250, 500]), is_train=True)

    if not arguments.only_specific:
        # preference/ with U walls
        for i in range(5001, 5501):
            create_arena_pref1(
                arguments.target_dir if arguments.unify
                else "{}/preferences/".format(arguments.target_dir),
                'c2_{}'.format(str(i).zfill(4)),
                time=random.choice([1000]),
                is_train=True)

    if not arguments.only_specific:
        # c2
        for i in range(1, 201):
            create_c2_arena(
                arguments.target_dir if arguments.unify
                else "{}/preferences/".format(arguments.target_dir),
                'c2_{}'.format(str(i).zfill(4)),
                max_reward=float(np.random.randint(5, 10)),
                time=random.choice([250, 500]), is_train=True,
                max_num_good_goals=np.random.randint(1, 2))

    print("Creating navigation arenas...")

    if not arguments.only_specific:
        # c3
        for i in range(1, 201):
            create_c3_arena(
                arguments.target_dir if arguments.unify
                else "{}/navigate/".format(arguments.target_dir),
                'c3_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]), is_train=True)

    if not arguments.only_specific:
        # c3
        for i in range(501, 701):
            create_c3_arena_basic(
                arguments.target_dir if arguments.unify
                else "{}/navigate/".format(arguments.target_dir),
                'c3_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]),
                num_walls=np.random.randint(5, 15),
                is_train=True)

    # cross
    for i in range(1, 200):
        create_arena_cross(
            arguments.target_dir if arguments.unify
            else "{}/navigate/".format(
                arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]), is_train=True)

    print("Creating zones arenas...")

    if not arguments.only_specific:
        # c4
        for i in range(1, 501):
            create_c4_arena(
                arguments.target_dir if arguments.unify
                else "{}/zones/".format(arguments.target_dir),
                'c4_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]),
                num_red_zones=8, max_orange_zones=3, is_train=True)

    print("Creating ramps arenas...")

    # c5
    for i in range(1, 200):
        create_c5_arena(
            arguments.target_dir if arguments.unify
            else "{}/ramps/".format(arguments.target_dir),
            'c5_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]), is_train=True)

    print("Creating generalize arenas...")

    if not arguments.only_specific:
        # c6
        for i in range(1, 201):
            create_c6_arena(
                arguments.target_dir if arguments.unify
                else "{}/generalize/".format(arguments.target_dir),
                'c6_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]), is_train=True)

    if not arguments.only_specific:
        # c6
        for i in range(501, 701):
            create_c6_arena_basic(
                arguments.target_dir if arguments.unify
                else "{}/generalize/".format(arguments.target_dir),
                'c6_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]), is_train=True,
                num_walls=np.random.randint(5, 15))

    print("Creating internal model arenas...")

    if not arguments.only_specific:
        # c7
        for i in range(1, 301):
            create_c7_arena(
                arguments.target_dir if arguments.unify
                else "{}/internal_model/".format(arguments.target_dir),
                'c7_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]), is_train=True)

    if not arguments.only_specific:
        # light 10 frames then black, transparent walls
        for i in range(5501, 6001):
            create_blackout_test_1(
                arguments.target_dir if arguments.unify
                else "{}/internal_model/".format(arguments.target_dir),
                'c7_{}'.format(str(i).zfill(4)),
                time=random.choice([1000]),
                is_train=True)

    print("Creating mazes arenas...")
    for i in range(1, 200):
        create_maze(
            arguments.target_dir if arguments.unify
            else "{}/mazes/".format(arguments.target_dir),
            'c8_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]), num_cells=2,
            obj=random.choice(['CylinderTunnel', 'door', 'Cardbox1']),
            is_train=True)

    print("Creating choices arenas...")

    if not arguments.only_specific:
        # choice
        for i in range(1, 400):
            create_arena_choice(
                arguments.target_dir if arguments.unify
                else "{}/choices/".format(arguments.target_dir),
                'c9_{}'.format(str(i).zfill(4)),
                time=random.choice([500, 1000]), is_train=True)

    print("Creating push arenas...")

    # push1
    for i in range(501, 701):
        create_arena_push1(
            arguments.target_dir if arguments.unify
            else "{}/push/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    # push2
    for i in range(1001, 1201):
        create_arena_push2(
            arguments.target_dir if arguments.unify
            else "{}/push/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    print("Creating tunnel arenas...")

    # tunnel1
    for i in range(1501, 1701):
        create_arena_tunnel1(
            arguments.target_dir if arguments.unify
            else "{}/tunnels/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)
    # tunnel2
    for i in range(1701, 1901):
        create_arena_tunnel2(
            arguments.target_dir if arguments.unify
            else "{}/tunnels/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    print("Creating ramp arenas...")

    # ramp1
    for i in range(1901, 2001):
        create_arena_ramp1(
            arguments.target_dir if arguments.unify
            else "{}/ramps/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)
    # ramp2
    for i in range(2101, 2201):
        create_arena_ramp2(
            arguments.target_dir if arguments.unify
            else "{}/ramps/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)
    # ramp3
    for i in range(2301, 2401):
        create_arena_ramp3(
            arguments.target_dir if arguments.unify
            else "{}/ramps/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    print("Creating narrow spaces arenas...")

    # narrow1
    for i in range(2501, 2701):
        create_arena_narrow_spaces_1(
            arguments.target_dir if arguments.unify
            else "{}/narrow/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    # narrow1
    for i in range(2701, 2901):
        create_arena_narrow_spaces_2(
            arguments.target_dir if arguments.unify
            else "{}/narrow/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    print("Creating reasoning step arenas...")

    # reasoning_step_1
    for i in range(5001, 5501):
        create_reasoning_step_1(
            arguments.target_dir if arguments.unify
            else "{}/reasoning/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    # reasoning_step_2
    for i in range(5501, 6001):
        create_reasoning_step_2(
            arguments.target_dir if arguments.unify
            else "{}/reasoning/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    # reasoning_step_3
    for i in range(6001, 6501):
        create_reasoning_step_3(
            arguments.target_dir if arguments.unify
            else "{}/reasoning/".format(arguments.target_dir),
            'c10_{}'.format(str(i).zfill(4)),
            time=random.choice([500, 1000]),
            is_train=True)

    # make fall 1
    for i in range(1, 100):
        create_make_fall_1(arguments.target_dir if arguments.unify
                           else "{}/make_fall/".format(
            arguments.target_dir),
                           'make_fall_1_{}'.format(str(i).zfill(4)),
                           time=random.choice([1000]), is_train=True)

    if arguments.phases:

        # create folders
        if not arguments.unify:
            for name_phase in ["choices_1", "choices_2", "choices_3",
                               "preferences_1", "preferences_2", "preferences_3"]:
                os.mkdir(
                    "{}/train_in_phases/{}".format(
                        arguments.target_dir, name_phase))

        print("Creating arenas to train in phases (choices)...")

        # choice 2 rewards different sizes, learn preferences
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_arena_choice_2(arguments.target_dir if arguments.unify
                                  else "{}/train_in_phases/choices_1/".format(
                arguments.target_dir),
                                  'c9_{}'.format(str(i).zfill(4)),
                                  time=random.choice([500, 1000]), is_train=True,
                                  rew_range=reward_range)

        # choice 2 rewards all sizes together
        for i in range(1, 25):
            reward_range_list = [[0.5, 5]]
            reward_range = random.choice(reward_range_list)
            create_arena_choice_2(arguments.target_dir if arguments.unify
                                  else "{}/train_in_phases/choices_2/".format(
                arguments.target_dir),
                                  'c9_{}'.format(str(i).zfill(4)),
                                  time=random.choice([500, 1000]), is_train=True,
                                  rew_range=reward_range)

        # choice 2 rewards only big balls, small difference
        for i in range(1, 25):
            reward_range_list = [[4, 5]]
            reward_range = random.choice(reward_range_list)
            create_arena_choice_2(arguments.target_dir if arguments.unify
                                  else "{}/train_in_phases/choices_3/".format(
                arguments.target_dir),
                                  'c9_{}'.format(str(i).zfill(4)),
                                  time=random.choice([500, 1000]), is_train=True,
                                  rew_range=reward_range)

        print("Creating arenas to train in phases (preferences)...")

        # left right
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_left_right(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_1/".format(
                arguments.target_dir),
                              'left_right_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)


        # front back
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_front_back(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_1/".format(
                arguments.target_dir),
                              'front_back_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)


        # cross
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_cross_green(arguments.target_dir if arguments.unify
                               else "{}/train_in_phases/preferences_1/".format(
                arguments.target_dir),
                               'cross_{}'.format(str(i).zfill(4)),
                               time=random.choice([500, 1000]), is_train=True,
                               rew_range=reward_range)

        # corners
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_corners_green(arguments.target_dir if arguments.unify
                                 else "{}/train_in_phases/preferences_1".format(
                arguments.target_dir),
                                 'corners_{}'.format(str(i).zfill(4)),
                                 time=random.choice([500, 1000]), is_train=True,
                                 rew_range=reward_range)

        # in front
        for i in range(1, 25):
            reward_range_list = [[0.5, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
            reward_range = random.choice(reward_range_list)
            create_in_front(arguments.target_dir if arguments.unify
                            else "{}/train_in_phases/preferences_1/".format(
                arguments.target_dir),
                            'in_front_{}'.format(str(i).zfill(4)),
                            time=random.choice([500, 1000]), is_train=True,
                            rew_range=reward_range)

        # left right
        for i in range(1, 25):
            reward_range_list = [[0.5, 5]]
            reward_range = random.choice(reward_range_list)
            create_left_right(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_2/".format(
                arguments.target_dir),
                              'left_right_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)

        # cross
        for i in range(1, 25):
            reward_range_list = [[0.5, 5]]
            reward_range = random.choice(reward_range_list)
            create_cross_green(arguments.target_dir if arguments.unify
                               else "{}/train_in_phases/preferences_2/".format(
                arguments.target_dir),
                               'cross_{}'.format(str(i).zfill(4)),
                               time=random.choice([500, 1000]), is_train=True,
                               rew_range=reward_range)

        # corners
        for i in range(1, 25):
            reward_range_list = [[0.5, 5]]
            reward_range = random.choice(reward_range_list)
            create_corners_green(arguments.target_dir if arguments.unify
                                 else "{}/train_in_phases/preferences_2/".format(
                arguments.target_dir),
                                 'corners_{}'.format(str(i).zfill(4)),
                                 time=random.choice([500, 1000]), is_train=True,
                                 rew_range=reward_range)

        # corners
        for i in range(1, 25):
            reward_range_list = [[4, 5]]
            reward_range = random.choice(reward_range_list)
            create_corners_green(arguments.target_dir if arguments.unify
                                 else "{}/train_in_phases/preferences_3/".format(
                arguments.target_dir),
                                 'corners_{}'.format(str(i).zfill(4)),
                                 time=random.choice([500, 1000]), is_train=True,
                                 rew_range=reward_range)

        # cross
        for i in range(1, 25):
            reward_range_list = [[4, 5]]
            reward_range = random.choice(reward_range_list)
            create_cross_green(arguments.target_dir if arguments.unify
                               else "{}/train_in_phases/preferences_3/".format(
                arguments.target_dir),
                               'cross_{}'.format(str(i).zfill(4)),
                               time=random.choice([500, 1000]), is_train=True,
                               rew_range=reward_range)

        # front back
        for i in range(1, 25):
            reward_range_list = [[0.5, 5]]
            reward_range = random.choice(reward_range_list)
            create_front_back(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_3/".format(
                arguments.target_dir),
                              'front_back_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)

        # front back
        for i in range(1, 25):
            reward_range_list = [[4, 5]]
            reward_range = random.choice(reward_range_list)
            create_front_back(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_3/".format(
                arguments.target_dir),
                              'front_back_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)

        # left right
        for i in range(1, 25):
            reward_range_list = [[4, 5]]
            reward_range = random.choice(reward_range_list)
            create_left_right(arguments.target_dir if arguments.unify
                              else "{}/train_in_phases/preferences_3/".format(
                arguments.target_dir),
                              'left_right_{}'.format(str(i).zfill(4)),
                              time=random.choice([500, 1000]), is_train=True,
                              rew_range=reward_range)
