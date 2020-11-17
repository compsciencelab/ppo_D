""" Create sets C1 to C10 of test arenas with known max reward for testing. """

import random
import numpy as np
from .edit_arenas import (add_object, write_arena, create_wall)
from .sample_features import (random_size, random_pos, random_rotation)
from .edit_arenas import (add_ramp_scenario, add_walled, add_choice,
                          cross_test, ramp_test_1, ramp_test_2, ramp_test_3,
                          tunnel_test_1, tunnel_test_2, push_test_1,
                          push_test_2, narrow_spaces_1, narrow_spaces_2,
                          preference_test_1, blackout_test_1, reasoning_step_1,
                          reasoning_step_2, reasoning_step_3)

objects_dict = {
    'reward_objects': [
        'GoodGoal',
        'GoodGoalBounce',
        'BadGoal',
        'BadGoalBounce',
        'GoodGoalMulti',
        'GoodGoalMultiBounce'
    ],
    'immovable_objects': [
        'Wall',
        'Ramp',
        'CylinderTunnel',
        'WallTransparent',
        'CylinderTunnelTransparent'],
    'movable_objects': [
        'Cardbox1',
        'Cardbox2',
        'UObject',
        'LObject',
        'LObject2'
    ],
    'zone_objects': [
        'DeathZone',
        'HotZone'
    ],
}


def create_c1_arena(target_path, arena_name, max_reward=5, time=250,
                    max_num_good_goals=1, is_train=False):
    """
    Create .yaml file for C1-type arena.
         - Only goals.
         - Fixed random size for all goals.
         - At least one green ball.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
        time (int): episode length.
        max_num_good_goals: goal limit.
   """

    allowed_objects = objects_dict['reward_objects']
    size_goal = (
    np.clip(random_size('GoodGoal')[0], 1.0, max_reward), 0.0, 0.0)
    position_goal = random_pos() if not is_train else None
    reward = float(max_reward)
    arena = add_object('', 'GoodGoal', size=size_goal, pos=position_goal)

    num_goals = 1
    worst_goal = 0.0
    min_reward = 0.0
    best_goal = size_goal[0]

    while reward - best_goal > size_goal[0]:

        category = allowed_objects[np.random.randint(0, len(allowed_objects))]
        position_goal = random_pos() if not is_train else None

        if category in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            reward -= size_goal[0]

        if category in ['BadGoal', 'BadGoalBounce']:
            worst_goal = min(worst_goal, size_goal[0])

        if category in ['GoodGoal', 'GoodGoalBounce']:
            best_goal = max(best_goal, size_goal[0])
            if num_goals >= max_num_good_goals:
                continue
            num_goals += 1

        arena = add_object(arena, category, size=size_goal, pos=position_goal)

    min_reward -= worst_goal

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c1', position_agent, rotation_agent


def create_c1_arena_weird(target_path, arena_name, time=250, is_train=False):
    """
    Create .yaml file for C1-type arena.
         - Only goals.
         - Fixed random size for all goals.
         - At least one green ball.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
        time (int): episode length.
        max_num_good_goals: goal limit.
   """

    position_goal = random_pos() if not is_train else None
    arena = add_object('', 'BadGoal', size=(0.5, 0.5, 0.5), pos=position_goal)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c1_weird', position_agent, rotation_agent


def create_c2_arena(target_path, arena_name, max_reward=5, time=250,
                    max_num_good_goals=1, is_train=False):
    """
    Create .yaml file for C2-type arena.
         - Only goals.
         - Different random size for all goals.
         - At least one green and one red ball.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
        time (int): episode length.
        max_num_good_goals: goal limit.
   """

    allowed_objects = objects_dict['reward_objects']
    reward = float(max_reward)
    size_goal = (
        np.clip(random_size('GoodGoal')[0], 1.0, max_reward), 0.0, 0.0)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', 'GoodGoal', size=size_goal, pos=position_goal)
    best_goal = size_goal[0]

    size_goal = random_size('BadGoal')
    position_goal = random_pos() if not is_train else None
    arena = add_object(arena, 'BadGoal', size=size_goal, pos=position_goal)
    worst_goal = size_goal[0]

    num_goals = 1
    min_reward = 0.0

    while reward - best_goal > size_goal[0]:

        category = allowed_objects[np.random.randint(0, len(allowed_objects))]
        size_goal = random_size(category)
        position_goal = random_pos() if not is_train else None

        if category in ['GoodGoalMulti', 'GoodGoalMultiBounce']:
            reward -= size_goal[0]

        if category in ['BadGoal', 'BadGoalBounce']:
            worst_goal = min(worst_goal, size_goal[0])

        if category in ['GoodGoal', 'GoodGoalBounce']:
            if num_goals >= max_num_good_goals:
                continue
            best_goal = max(best_goal, size_goal[0])
            num_goals += 1

        arena = add_object(arena, category, size=size_goal, pos=position_goal)

    min_reward -= worst_goal

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c2', position_agent, rotation_agent


def create_c3_arena_basic(target_path, arena_name, time=250, num_walls=1,
                          is_train=False):
    """ include explanation """

    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])
    size_goal = random_size(category)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', category, size=size_goal, pos=position_goal)

    arena = add_walled(arena, num_walls=num_walls)
    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c3', position_agent, rotation_agent


def create_c3_arena(target_path, arena_name, time=250, max_movable=3,
                    max_immovable=3, is_train=False):
    """
    Create .yaml file for C3-type arena.
        - One random positive reward ball, random sized
        - With probability 0.5 add red ball, random sized
        - Create a wall maze by randomly spawning between 1 and 10 walls
        - If specified randomly add multiple movable and immovable objects
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        max_reward (float): set max reward for arena. Relates to arena complexity.
        time (int): episode length.
        max_movable (int): set a limit to number of movable objects.
        max_immovable (int): set a limit to number of immovable.
    """

    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])

    size_goal = random_size(category)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', category, size=size_goal, pos=position_goal)

    if random.random() > 0.5:
        category = random.choice(['BadGoal', 'BadGoalBounce'])
        size_goal = random_size(category)
        position_goal = random_pos() if not is_train else None
        arena = add_object(arena, category, size=size_goal, pos=position_goal)

    for _ in range(max_movable):
        if random.random() > 0.1:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    for _ in range(max_immovable):
        if random.random() > 0.1:
            category = random.choice(objects_dict['immovable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c3', position_agent, rotation_agent


def create_c4_arena(target_path, arena_name, time=250, num_red_zones=2,
                    max_orange_zones=1, max_movable=1, max_immovable=1,
                    is_train=False):
    """
    Create .yaml file for C4-type arena.
        - 1 green food (stationary) and some red zones
        - add orange ball with probability 0.5
        - add orange zone with probability 0.5
        - add immobable object with probability 0.1
        - add movable object with probability 0.1
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        time (int): episode length.
        num_red_zones (int): fixed number of red zones.
        max_orange_zones (int): set a limit to number of orange zones.
        max_movable (int): set a limit to number of movable objects.
        max_immovable (int): set a limit to number of immovable.
    """

    size_goal = random_size('GoodGoal')
    position_goal = random_pos() if not is_train else None
    arena = add_object('', 'GoodGoal', size=size_goal, pos=position_goal)

    if random.random() > 0.5:
        size_goal = random_size('GoodGoalMulti')
        position_goal = random_pos() if not is_train else None
        arena = add_object(arena, 'GoodGoalMulti', size=size_goal,
                           pos=position_goal)

    for _ in range(num_red_zones):
        size_object = random_size('DeathZone')
        pos_object = random_pos() if not is_train else None
        arena = add_object(arena, 'DeathZone', size=size_object,
                           pos=pos_object)

    for _ in range(max_orange_zones):
        if random.random() > 0.5:
            size_object = random_size('HotZone')
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, 'HotZone', size=size_object,
                               pos=pos_object)

    for _ in range(max_movable):
        if random.random() > 0.8:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    for _ in range(max_immovable):
        if random.random() > 0.8:
            category = random.choice(objects_dict['immovable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c4', position_agent, rotation_agent


def create_c5_arena(target_path, arena_name, time=250, max_movable=1,
                    max_immovable=1, is_train=False):
    """
    Create .yaml file for C5-type arena.
        - from 1 to 2 platforms accessible by ramps with a goal on top.
        - if specified, add multiple movable and immovable objects.
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        time (int): episode length.
        max_movable (int): set a limit to number of movable objects.
        max_immovable (int): set a limit to number of immovable.
    """

    arena = add_ramp_scenario('')
    arena = add_ramp_scenario(arena)

    for _ in range(max_movable):
        if random.random() < 0.8:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    for _ in range(max_immovable):
        if random.random() < 0.8:
            category = random.choice(['Wall', 'Ramp', 'CylinderTunnel'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c5', position_agent, rotation_agent


def create_c6_arena_basic(target_path, arena_name, time=250, num_walls=1,
                          is_train=False):
    """ include explanation """

    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])
    size_goal = random_size(category)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', category, size=size_goal, pos=position_goal)
    arena = add_walled(arena, num_walls=num_walls, random_rgb=True)
    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c6', position_agent, rotation_agent


def create_c6_arena(target_path, arena_name, time=250, max_movable=3,
                    max_immovable=3, is_train=False):
    """
    Create .yaml file for C6-type arena.
        - One random positive reward ball, random sized
        - add a second positive reward , with probability 0.5
        - add up to 2 red balls , with probability 0.5 each
        - Add multiple (1 to 10) walls with random color.
        - If specifiedm add also extra multiple movable and immovable objects (with random color)
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        time (int): episode length.
        max_movable (int): set a limit to number of movable objects.
        max_immovable (int): set a limit to number of immovable.
    """

    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])
    size_goal = random_size(category)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', category, size=size_goal, pos=position_goal)

    if random.random() > 0.5:
        category = random.choice(
            ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti',
             'GoodGoalMultiBounce'])
        size_goal = random_size(category)
        position_goal = random_pos() if not is_train else None
        arena = add_object(arena, category, size=size_goal, pos=position_goal)

    for _ in range(2):
        if random.random() > 0.5:
            category = random.choice(['BadGoal', 'BadGoalBounce'])
            size_goal = random_size(category)
            position_goal = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_goal,
                               pos=position_goal)

    for _ in range(max_movable):
        if random.random() < 0.8:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    for _ in range(max_immovable):
        if random.random() < 0.8:
            category = random.choice(['Wall', 'Ramp', 'CylinderTunnel'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'c6', position_agent, rotation_agent


def create_c7_arena(target_path, arena_name, time=250, max_movable=3,
                    max_immovable=3, is_train=False):
    """
    Create .yaml file for C7-type arena.
        - One random positive reward ball, random sized
        - add a second positive rewards , with probability 0.5
        - add up to 2 red balls , with probability 0.5 each
        - With probability 0.5 add red balls, random sized
        - Add multiple movable and immovable objects
        - random blackouts
    Parameters:
        target_path (str): save dir path.
        arena_name (str): save name arena.
        time (int): episode length.
        max_movable (int): set a limit to number of movable objects.
        max_immovable (int): set a limit to number of immovable.
    """

    blackout_options = [[-20], [-40], [-60], [25, 30, 50, 55, 75],
                        [50, 55, 75, 80, 100, 105, 125]]
    category = random.choice(
        ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce'])

    size_goal = random_size(category)
    position_goal = random_pos() if not is_train else None
    arena = add_object('', category, size=size_goal, pos=position_goal)

    if random.random() > 0.5:
        category = random.choice(
            ['GoodGoal', 'GoodGoalBounce', 'GoodGoalMulti',
             'GoodGoalMultiBounce'])
        size_goal = random_size(category)
        position_goal = random_pos() if not is_train else None
        arena = add_object(arena, category, size=size_goal, pos=position_goal)

    for _ in range(2):
        if random.random() > 0.5:
            category = random.choice(['BadGoal', 'BadGoalBounce'])
            size_goal = random_size(category)
            position_goal = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_goal,
                               pos=position_goal)

    for _ in range(max_movable):
        if random.random() < 0.8:
            category = random.choice(objects_dict['movable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    for _ in range(max_immovable):
        if random.random() < 0.8:
            category = random.choice(objects_dict['immovable_objects'])
            size_object = random_size(category) if not is_train else None
            pos_object = random_pos() if not is_train else None
            arena = add_object(arena, category, size=size_object,
                               pos=pos_object)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options))

    return 'c7', position_agent, rotation_agent


def create_maze(target_path, arena_name, time=250, num_cells=3, obj=None,
                is_train=False):
    """ include explanation """

    wall_type = random.choice(['Wall', 'WallTransparent'])
    arena = ''

    if obj == 'CylinderTunnel':
        gap = 3
    else:
        gap = 2

    num_cells_x = num_cells
    num_cells_y = num_cells

    side_wall_len_x = int(40 / num_cells_x)
    side_wall_len_y = int(40 / num_cells_y)

    location_pillars_x = list(range(0, 40, side_wall_len_x))[1:]
    location_pillars_y = list(range(0, 40, side_wall_len_y))[1:]

    walls_loc_x = location_pillars_x
    walls_loc_x.append(40)

    walls_loc_y = location_pillars_y
    walls_loc_y.append(40)

    prev_y = 0
    prev_x = 0
    z_size = random.choice([0.5, 10])

    for y in walls_loc_y:
        for x in walls_loc_x:

            size_1, pos_1, size_2, pos_2 = create_wall(
                (prev_x, y), (x, y), z_size, obj='door', gap=gap)

            arena = add_object(arena, wall_type, size=size_1, pos=pos_1, rot=0)
            arena = add_object(arena, wall_type, size=size_2, pos=pos_2, rot=0)

            if obj != 'door':
                size, pos = create_wall(
                    (prev_x, y), (x, y), z_size, obj=obj, gap=gap)
                arena = add_object(arena, obj, size=size, pos=pos, rot=0)

            size_1, pos_1, size_2, pos_2 = create_wall(
                (x, prev_y), (x, y), z_size, obj='door', gap=gap)
            arena = add_object(arena, wall_type, size=size_1, pos=pos_1, rot=0)
            arena = add_object(arena, wall_type, size=size_2, pos=pos_2, rot=0)

            if obj != 'door':
                size, pos = create_wall(
                    (x, prev_y), (x, y), z_size, obj=obj, gap=gap)
                arena = add_object(arena, obj, size=size, pos=pos, rot=90)
            prev_x = x

        prev_x = 0
        prev_y = y

    size_goal = random_size('GoodGoal')
    position_goal = random_pos() if not is_train else None
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    position_agent = random_pos() if not is_train else None
    rotation_agent = random_rotation() if not is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'maze', position_agent, rotation_agent


def create_arena_choice(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena = add_choice('')

    size_goal = random_size('GoodGoal')
    position_goal = random_pos() if not is_train else None
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size_goal = random_size('GoodGoalMulti')
    position_goal = random_pos() if not is_train else None
    arena = add_object(arena, 'GoodGoalMulti', size=size_goal,
                       pos=position_goal)

    save_name = '{}/{}'.format(target_path, arena_name, is_train=is_train)
    write_arena(save_name, time, arena)

    return 'choice', (20, 0, 20), 0


def create_arena_cross(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = cross_test("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'cross', pos_agent, rot_agent


def create_arena_push1(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = push_test_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'push1', pos_agent, rot_agent


def create_arena_push2(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = push_test_2("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'push2', pos_agent, rot_agent


def create_arena_tunnel1(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = tunnel_test_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'tunnel1', pos_agent, rot_agent


def create_arena_tunnel2(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = tunnel_test_2("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'tunnel2', pos_agent, rot_agent


def create_arena_ramp1(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = ramp_test_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'ramp1', pos_agent, rot_agent


def create_arena_ramp2(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = ramp_test_2("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'ramp2', pos_agent, rot_agent


def create_arena_ramp3(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = ramp_test_3("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'ramp3', pos_agent, rot_agent


def create_arena_narrow_spaces_1(target_path, arena_name, time=250,
                                 is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = narrow_spaces_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'narrow1', pos_agent, rot_agent


def create_arena_narrow_spaces_2(target_path, arena_name, time=250,
                                 is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = narrow_spaces_2("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'narrow2', pos_agent, rot_agent


def create_arena_pref1(target_path, arena_name, time=250, is_train=False):
    """ include explanation """

    arena, pos_agent, rot_agent = preference_test_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'pref', pos_agent, rot_agent


def create_blackout_test_1(target_path, arena_name, time=250, is_train=False):
    arena, pos_agent, rot_agent = blackout_test_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena, blackouts=[10, 1000])

    return 'blackout', pos_agent, rot_agent


def create_reasoning_step_1(target_path, arena_name, time=250, is_train=False):
    arena, pos_agent, rot_agent = reasoning_step_1("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'reasoning1', pos_agent, rot_agent


def create_reasoning_step_2(target_path, arena_name, time=250, is_train=False):
    arena, pos_agent, rot_agent = reasoning_step_2("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'reasoning2', pos_agent, rot_agent


def create_reasoning_step_3(target_path, arena_name, time=250, is_train=False):
    arena, pos_agent, rot_agent = reasoning_step_3("", is_train=is_train)
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'reasoning3', pos_agent, rot_agent


def create_front_back(target_path, arena_name, time=1000, is_train=False,
                      rew_range=[0.5, 5]):
    """ include explanation """

    arena = ''

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 2)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (20, 0, 30)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (20, 0, 10)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    pos_agent = (20, 0, 20)
    arena = add_object(arena, "Agent", pos=pos_agent, rot=0)

    save_name = '{}/{}'.format(target_path, arena_name)
    blackout_options_2 = [[-10], [1000, 1000]]
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options_2))

    return 'front_back', (20, 0, 20), 0


def create_left_right(target_path, arena_name, time=1000, is_train=False,
                      rew_range=[0.5, 5]):
    """ include explanation """

    arena = ''

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 2)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (10, 0, 20)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (30, 0, 20)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    pos_agent = (20, 0, 20)
    arena = add_object(arena, "Agent", pos=pos_agent, rot=0)

    save_name = '{}/{}'.format(target_path, arena_name)
    blackout_options_2 = [[-10], [1000, 1000]]
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options_2))

    return 'left_right', (20, 0, 20), 0


def create_corners_green(target_path, arena_name, time=1000, is_train=False,
                         rew_range=[0.5, 5]):
    """ include explanation """

    arena = ''

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 4)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (37, 0, 37)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (3, 0, 37)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[2]
    size_goal = (size, size, size)
    position_goal = (37, 0, 3)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[3]
    size_goal = (size, size, size)
    position_goal = (3, 0, 3)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    pos_agent = (20, 0, 20)
    arena = add_object(arena, "Agent", pos=pos_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    blackout_options_2 = [[-10], [1000, 1000]]
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options_2))

    return 'corners_green', (20, 0, 20), 0


def create_cross_green(target_path, arena_name, time=1000, is_train=False,
                       rew_range=[0.5, 5]):
    """ include explanation """

    arena = ''

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 4)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (30, 0, 20)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (10, 0, 20)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[2]
    size_goal = (size, size, size)
    position_goal = (20, 0, 10)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[3]
    size_goal = (size, size, size)
    position_goal = (20, 0, 30)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    pos_agent = (20, 0, 20)
    arena = add_object(arena, "Agent", pos=pos_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    blackout_options_2 = [[-10], [1000, 1000]]
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options_2))

    return 'cross_green', (20, 0, 20), 0


def create_in_front(target_path, arena_name, time=1000, is_train=False,
                    rew_range=[0.5, 5]):
    """ include explanation """

    arena = ''

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 2)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (30, 0, 35)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (10, 0, 35)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    pos_agent = (20, 0, 3)
    arena = add_object(arena, "Agent", pos=pos_agent, rot=0)

    save_name = '{}/{}'.format(target_path, arena_name)
    blackout_options_2 = [[-10], [1000, 1000]]
    write_arena(save_name, time, arena,
                blackouts=random.choice(blackout_options_2))

    return 'in_front', (20, 0, 20), 0


def create_make_fall_1(target_path, arena_name, time=1000, is_train=False,
                       rew_range=[5, 50]):
    """
    The  reward is on top of a box. Agent must push the box to make reward fall down.
    """
    arena = ''

    height = random.randint(2, 7)
    pos_box = (random.randint(5, 35), 0, random.randint(5, 35))
    siz_box = (random.randint(2, 4), height, random.randint(2, 4))
    category = random.choice(['Cardbox1', 'Cardbox2'])
    arena = add_object(arena, category, size=siz_box, pos=pos_box, rot=0)

    rew_size = random.randint(1, 5)
    rew_pos = (pos_box[0], siz_box[1] + rew_size / 2, pos_box[2])
    arena = add_object(arena, random.choice(["GoodGoal", "GoodGoalMulti"]),
                       size=(rew_size, rew_size, rew_size), pos=rew_pos)

    rotation_agent = random_rotation()
    position_agent = random_pos()
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'make_fall', position_agent, rotation_agent


def create_arena_choice_2(target_path, arena_name, time=250, is_train=False,
                          rew_range=[0.5, 5]):
    """ include explanation """

    arena = add_choice('')

    range_list = list(np.linspace(rew_range[0], rew_range[1],
                                  (rew_range[1] - rew_range[0]) / 0.1 + 1))
    range_list = [round(elem, 2) for elem in range_list]
    list_siz = random.sample(range_list, 4)

    size = list_siz[0]
    size_goal = (size, size, size)
    position_goal = (30, 0, 30)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[1]
    size_goal = (size, size, size)
    position_goal = (10, 0, 30)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = list_siz[2]
    size_goal = (size, size, size)
    position_goal = (10, 0, 10)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    size = size = list_siz[3]
    size_goal = (size, size, size)
    position_goal = (30, 0, 10)
    arena = add_object(arena, 'GoodGoal', size=size_goal, pos=position_goal)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'choice2', (20, 0, 20), 0


def create_box_reasoning(target_path, arena_name, time=1000, is_train=True):
    """ The  reward is on top of a box. agent must push box in the right position to be able to acces the reward platform."""
    arena = ''
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(13.5, 0, 20), rot=90)
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(22, 0, 20), rot=90)
    arena = add_object(arena, 'Cardbox1', size=(4, 2, 4), pos=(17.75, 0, 9), rot=90)
    arena = add_object(arena, 'Ramp', size=(8, 2, 8), pos=(28, 0, 20), rot=90)
    arena = add_object(arena, 'GoodGoal', size=(3, 3, 3), pos=(13.5, 2.25, 20), rot=0)
    

    agent_rotation = random_rotation()
    while True:

        agent_pos = random_pos()

        if pos_on_obj(agent_pos,(13.5, 0, 20),(4, 2, 4)) == False:
            if pos_on_obj(agent_pos,(22, 0, 20),(4, 2, 4)) == False:
                if pos_on_obj(agent_pos,(17.75, 0, 9),(4, 2, 4)) == False:
                    if pos_on_obj(agent_pos,(28, 0, 20),(8, 2, 8)) == False:
                        break

    
    arena = add_object(arena, "Agent", pos= agent_pos, rot=agent_rotation)                     
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'box_reasoning', (20, 0, 3), 0



def create_box_reasoning_hard(target_path, arena_name, time=1000, is_train=True):
    """ The  reward is on top of a box. agent must push box in the right position to be able to acces the reward platform."""
    pos_box_ = [(10, 0, 10),(10, 0, 30),(30, 0, 10),(30, 0, 30)]
    pos_box = random.choice(pos_box_)
    arena = ''
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(13.5, 0, 20), rot=90)
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(22, 0, 20), rot=90)
    arena = add_object(arena, 'Cardbox1', size=(4, 2, 4), pos=pos_box, rot=90)
    arena = add_object(arena, 'Ramp', size=(8, 2, 8), pos=(28, 0, 20), rot=90)
    arena = add_object(arena, 'GoodGoal', size=(3, 3, 3), pos=(13.5, 2.25, 20), rot=0)
    

    agent_rotation = random_rotation()
    while True:

        agent_pos = random_pos()

        if pos_on_obj(agent_pos,(13.5, 0, 20),(4, 2, 4)) == False:
            if pos_on_obj(agent_pos,(22, 0, 20),(4, 2, 4)) == False:
                if pos_on_obj(agent_pos, pos_box,(4, 2, 4)) == False:
                    if pos_on_obj(agent_pos,(28, 0, 20),(8, 2, 8)) == False:
                        break

    
    arena = add_object(arena, "Agent", pos= agent_pos, rot=agent_rotation)                     
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'box_reasoning', (20, 0, 3), 0


def create_box_reasoning_2(target_path, arena_name, time=1000, is_train=True):
    """ The  reward is on top of a box. agent must push box in the right position to be able to acces the reward platform."""
    pos_box_ = [(10, 0, 10),(10, 0, 30),(30, 0, 10),(30, 0, 30)]
    pos_box1, pos_box2  = random.sample(pos_box_, k=2)

    arena = ''
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(7.5, 0, 20), rot=90)
    arena = add_object(arena, 'WallTransparent', size=(4, 2, 4), pos=(22, 0, 20), rot=90)
    arena = add_object(arena, 'Cardbox1', size=(4, 2, 4), pos=pos_box1, rot=90)
    arena = add_object(arena, 'Cardbox1', size=(4, 2, 4), pos=pos_box2, rot=90)
    arena = add_object(arena, 'Ramp', size=(8, 2, 8), pos=(28, 0, 20), rot=90)
    arena = add_object(arena, 'GoodGoal', size=(3, 3, 3), pos=(7.5, 2.25, 20), rot=0)
    

    agent_rotation = random_rotation()
    while True:

        agent_pos = random_pos()

        if pos_on_obj(agent_pos,(7.5, 0, 20),(4, 2, 4)) == False:
            if pos_on_obj(agent_pos,(22, 0, 20),(4, 2, 4)) == False:
                if pos_on_obj(agent_pos, pos_box1,(4, 2, 4)) == False:
                    if pos_on_obj(agent_pos, pos_box2,(4, 2, 4)) == False:
                        if pos_on_obj(agent_pos,(28, 0, 20),(8, 2, 8)) == False:
                            break

    
    arena = add_object(arena, "Agent", pos= agent_pos, rot=agent_rotation)                     
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'box_reasoning', (20, 0, 3), 0


def create_push_down(target_path, arena_name, time=1000, is_train=True):
    """ The  reward is on top of a box. agent must push box in the right position to be able to acces the reward platform."""


    coin = random.choice([0,1])
    if coin == 1:
        box_pos = (random.randint(25, 140) / 10., 1., random.randint(25, 200) / 10.)
        fake_box_pos = (random.randint(260, 380) / 10., 1., random.randint(25, 200) / 10.)
    else:
        fake_box_pos = (random.randint(25, 140) / 10., 1., random.randint(25, 200) / 10.)
        box_pos = (random.randint(260, 380) / 10., 1., random.randint(25, 200) / 10.)

    pos_goal = (random.randint(10, 390) / 10., 1., random.randint(250, 390) / 10.)
    goal_wal_pos = (pos_goal[0], 0,  pos_goal[2])
    

    arena = ''
    arena = add_object(arena, 'Wall', size=(0.3, 1, 0.3), pos=goal_wal_pos, rot=0, RGB= (153, 153, 153))
    arena = add_object(arena, 'Wall', size=(40, 1, 20), pos=(20, 0, 10), rot=0, RGB= (0, 0, 255))
    arena = add_object(arena, 'Wall', size=(11, 2, 5), pos=(20, 1, 2.5), rot=0, RGB= (0, 0, 255))
    arena = add_object(arena, 'Wall', size=(1, .5, 15), pos=(20, 1, 12.5), rot=0, RGB=(153, 153, 153))

    arena = add_object(arena, 'Cardbox2', size=(1.5, 1.5, 1.5), pos=box_pos, rot=0)
    arena = add_object(arena, 'Wall', size=(1.5, 1.5, 1.5), pos=fake_box_pos, rot=0, RGB=(153, 153, 153))
    arena = add_object(arena, 'GoodGoal', size=(1,1,1), pos=pos_goal, rot=0)
    arena = add_object(arena, "Agent", pos= (20, 3, 2.5), rot=2)   
              
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'push_down_decieve', (20, 0, 3), 0


def create_push_red(target_path, arena_name, time=1000, is_train=True):
    """ The  reward is on top of a box. agent must push box in the right position to be able to acces the reward platform."""
    arena = ''
    
    arena = add_object(arena, 'DeathZone', size=(10, 0, 10), pos=(20, 0, 20), rot=45)
    arena = add_object(arena, 'GoodGoal', size=(1, 1, 1), pos=(20, 0, 20), rot=0)
    
    
    while True:

        agent_pos = random_pos()
        pos_box1 = (random.randint(65, 335) / 10., 0., random.randint(65, 335) / 10.)
        if pos_on_obj(pos_box1,(20, 0, 20),(15, 0, 15)) == False:
            if pos_on_obj(agent_pos,(20, 0, 20),(15, 0, 15)) == False:
                if pos_on_obj(agent_pos, pos_box1,(6, .2, 2)) == False:
                    break

    agent_rotation = random_rotation()
    box_rotation = random_rotation()
    arena = add_object(arena, "Agent", pos= agent_pos, rot=agent_rotation)
    arena = add_object(arena, 'Cardbox2', size=(6, .2, 2), pos=pos_box1, rot=box_rotation)                  
    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena)

    return 'push_red', (20, 0, 3), 0



def pos_on_obj(agent_pos, obj_pos, obs_size):
    x_lim1 = obj_pos[0] -  obs_size[0]/2
    x_lim2 = obj_pos[0] +  obs_size[0]/2
    y_lim1 = obj_pos[2] -  obs_size[2]/2
    y_lim2 = obj_pos[2] +  obs_size[2]/2

    if (agent_pos[0] < x_lim1) or (agent_pos[0]> x_lim2):
        if (agent_pos[2] < y_lim1) or (agent_pos[2]> y_lim2):
            return False
        else:
            return True
    else:
        return True



