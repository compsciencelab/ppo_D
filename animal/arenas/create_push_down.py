""" Create a train set. """

import random
import numpy as np
from animal.arenas.utils import (
create_push_down
)

if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--target-dir', help='path to arenas train directory')

    arguments = parser.parse_args()
    """ if not os.path.isdir(arguments.target_dir):
        os.mkdir(arguments.target_dir) """


    """skills = ["box_reasoning"]



     for skill in skills:
        if not os.path.isdir("{}/{}".format(arguments.target_dir, skill)):
            os.mkdir("{}/{}".format(arguments.target_dir, skill)) """



    # box reasoning
    for i in range(1, 1000):
        reward_range_list = [[4, 5]]
        reward_range = random.choice(reward_range_list)
        create_push_down( "{}/".format(arguments.target_dir),
            'c2_{}'.format(str(i).zfill(4)))
