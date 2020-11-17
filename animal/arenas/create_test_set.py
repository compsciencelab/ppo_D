""" Create a basic test set. """

import random
import numpy as np
from utils import (
    create_c1_arena,
    create_c1_arena_weird,
    create_c2_arena,
    create_c3_arena_basic,
    create_c3_arena,
    create_c4_arena,
    create_c5_arena,
    create_c6_arena_basic,
    create_c6_arena,
    create_c7_arena,
    create_maze,
    create_mix_maze,
    create_arena_choice,
)

if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas test directory')

    arguments = parser.parse_args()
    if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir)
    #
    # # c1
    # for i in range(1, 11):
    #     create_c1_arena(arguments.target_dir, 'c1_{}'.format(
    #         str(i).zfill(4)), max_reward=float(np.random.randint(5, 8)))
    #
    # # c1_weird
    # for i in range(11, 21):
    #     create_c1_arena_weird(arguments.target_dir, 'c1_{}'.format(
    #         str(i).zfill(4)))
    #
    # # c2
    # for i in range(1, 11):
    #     create_c2_arena(arguments.target_dir, 'c2_{}'.format(
    #         str(i).zfill(4)), max_reward=5,
    #                     max_num_good_goals=np.random.randint(1, 2))
    #
    # # c3
    # for i in range(1, 11):
    #     create_c3_arena(arguments.target_dir, 'c3_{}'.format(
    #         str(i).zfill(4)), time=random.choice([250, 500, 1000]))
    #
    # # c3
    # for i in range(11, 21):
    #     create_c3_arena_basic(arguments.target_dir, 'c3_{}'.format(
    #         str(i).zfill(4)), time=random.choice([250, 500, 1000]),
    #                           num_walls=np.random.randint(5, 15))
    #
    # # c4
    # for i in range(1, 11):
    #     create_c4_arena(arguments.target_dir, 'c4_{}'.format(
    #         str(i).zfill(4)), time=random.choice([250, 500, 1000]),
    #                     num_red_zones=8, max_orange_zones=3)
    #
    # # c5
    # for i in range(1, 11):
    #     create_c5_arena(arguments.target_dir, 'c5_{}'.format(str(i).zfill(4)),
    #                     time=random.choice([250, 500, 1000]))
    #
    # # c6
    # for i in range(1, 11):
    #     create_c6_arena(arguments.target_dir, 'c6_{}'.format(str(i).zfill(4)),
    #                     time=random.choice([250, 500, 1000]))
    #
    # # c6
    # for i in range(11, 21):
    #     create_c6_arena_basic(arguments.target_dir, 'c6_{}'.format(str(i).zfill(4)),
    #                           time=random.choice([250, 500, 1000]),
    #                           num_walls=np.random.randint(5, 15))
    #
    # # c7
    # for i in range(1, 11):
    #     create_c7_arena(arguments.target_dir, 'c7_{}'.format(str(i).zfill(4)),
    #                     time=random.choice([250, 500, 1000]))

    # mazes
    for i in range(1, 11):
        create_maze(arguments.target_dir, 'c8_{}'.format(str(i).zfill(4)),
                    time=random.choice([250, 500, 1000]),
                    num_cells=np.random.randint(2, 5),
                    obj=random.choice(['CylinderTunnel', 'door', 'Cardbox1']))

    # choice
    for i in range(1, 11):
        create_arena_choice(arguments.target_dir, 'c9_{}'.format(str(i).zfill(4)),
                    time=random.choice([250, 500, 1000]),
                    is_train=True)

    # special mazes
    for i in range(1, 21):
        create_mix_maze(arguments.target_dir, 'c10_{}'.format(str(i).zfill(4)),
                    time=random.choice([250, 500, 1000]),
                    num_cells=np.random.randint(2, 7), max_movable =np.random.randint(2, 7) ,max_immovable = np.random.randint(2, 7),num_red_zones=np.random.randint(0, 3),max_orange_zones= np.random.randint(0, 3))