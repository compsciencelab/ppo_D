import random
import numpy as np
from animal.arenas.utils.sample_features import (
    random_size, sample_position_with_respect_to, random_color,
    random_rotation, random_pos)


def add_object(s, object_name, pos=None, size=None, RGB=None, rot=None):
    s += "    - !Item \n      name: {} \n".format(object_name)

    if RGB is None:
        if object_name is 'Ramp':
            RGB = (255, 0, 255)
        elif object_name is 'CylinderTunnel':
            RGB = (153, 153, 153)
        elif object_name is 'Wall':
            RGB = (153, 153, 153)

    if pos is not None:
        s += "      positions: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(
            pos[0], pos[1], pos[2])
    if size is not None:
        s += "      sizes: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(
            size[0], size[1], size[2])
    if RGB is not None:
        s += "      colors: \n      - !RGB {{r: {}, g: {}, b: {}}}\n".format(
            RGB[0], RGB[1], RGB[2])
    if rot is not None:
        s += "      rotations: [{}]\n".format(rot)
    return s


def write_arena(fname, time, arena_str, blackouts=None):
    blackouts_str = "blackouts: {} \n    ".format(
        blackouts) if blackouts else ""
    with open("{}.yaml".format(fname), 'w+') as f:
        f.write(
            "!ArenaConfig \narenas: \n  0: !Arena \n    t: {} \n    {}items:\n".format(
                time, blackouts_str))
        f.write(arena_str)


def add_ramp_scenario(arena, is_train=False):
    # create a wall as a platform
    category = np.random.choice(['Wall', 'Cardbox1', 'Cardbox2'])

    size_wall = (
        np.clip(random_size(category)[0], 2, 2),
        np.clip(random_size(category)[1], 2, 2),
        np.clip(random_size(category)[2], 2, 2)
    )

    rotation_wall = random.choice([0])
    position_wall = sample_position_with_respect_to((20, 0, 20), 'close')
    arena = add_object(arena, category, size=size_wall, pos=position_wall,
                       RGB=(0, 0, 255), rot=rotation_wall)

    # locate a reward on the wall
    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    size_object = random_size(category)
    pos_goal = (
        position_wall[0],
        size_wall[1] + size_object[1] + 0.1,
        position_wall[2]
    )

    arena = add_object(
        arena, category, size=size_object, RGB=(0, 0, 255), pos=pos_goal)

    pos_agent = sample_position_with_respect_to(pos_goal, random.choice(
        ['close', 'medium', 'far']))

    # create ramps to access the reward
    category = 'Ramp'

    # randomly choose how many ramps
    for _ in range(3):
        for rot, pos_shift, size, lol in zip(
                [0, 90, 180, 270],
                [(0.0, 0.5), (0.5, 0.0), (0.0, -0.5), (-0.5, 0.0)],
                [(0, 2), (2, 0), (0, 2), (2, 0)],
                [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]):

            size_object = (4, size_wall[1], 4)

            if random.random() > 0.0:
                position_object = (
                    position_wall[0] + (
                        size_wall[0] + size_object[size[0]]) * pos_shift[0],
                    0.0,
                    position_wall[2] + (
                        size_wall[2] + size_object[size[1]]) * pos_shift[1])

                arena = add_object(
                    arena, category, size=size_object, rot=rot,
                    pos=position_object)

    if random.random() < 0.1:
        pos_agent = position_object

    arena = add_object(
        arena, "Agent", pos=(pos_agent[0], size_wall[1] + 0.1, pos_agent[2]))

    return arena


def add_choice(arena, is_train=False):
    # create platform in the middle

    height = np.random.randint(1, 10)
    category = 'Wall'
    size_wall = (4, 1, 4)
    position_wall = (20., 0., 20.)
    rotation_wall = 0.
    arena = add_object(
        arena, category, size=size_wall, pos=position_wall,
        rot=rotation_wall, RGB=(0, 0, 255))

    # create walls to divide the arena
    size_wall = (0.5, height, 18)
    for position_wall, rotation_wall in zip(
            [(9., 0., 20.), (31., 0., 20.), (20., 0., 9), (20., 0., 31)],
            [90., 270., 0., 180.]):
        arena = add_object(
            arena, category, size=size_wall, pos=position_wall,
            rot=rotation_wall)

    # locate agent in the platform
    position_agent = (20., 5., 20.)
    arena = add_object(arena, "Agent", pos=position_agent)

    return arena


def add_walled(arena, num_walls=1, random_rgb=False):
    category = 'Wall'

    for _ in range(num_walls):
        position_wall = sample_position_with_respect_to((20, 0, 20), 'far')
        rotation_wall = random.choice([0, 90, 180, 360])
        size_wall = (0.5, 5, 10)
        arena = add_object(
            arena, category, size=size_wall, pos=position_wall,
            rot=rotation_wall, RGB=random_color() if random_rgb else None)

    return arena


def cross_test(arena, is_train=False):
    """
    The arena divided in 2 parts, the agent is required to find a hole in the
    wall to cross.
    """

    position_door_axis_1 = np.random.randint(2, 38)
    position_door_axis_2 = np.random.randint(10, 30)
    height_walls = random.choice([0.5, 2, 5, 10])

    size_wall_1 = (0.5, height_walls, - 1.15 + position_door_axis_1)
    size_wall_2 = (0.5, height_walls, 38.75 - position_door_axis_1)
    sizes_wall = [size_wall_1, size_wall_2]

    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])
    rotation_wall = random.choice([90, 0])

    if rotation_wall == 90:
        positions_wall = [(size_wall_1[-1] / 2, 0, position_door_axis_2),
                          (40 - size_wall_2[-1] / 2, 0, position_door_axis_2)]
    else:
        positions_wall = [(position_door_axis_2, 0, size_wall_1[-1] / 2),
                          (position_door_axis_2, 0, 40 - size_wall_2[-1] / 2)]

    for size_wall, position_wall in zip(sizes_wall, positions_wall):
        arena = add_object(
            arena, category, size=size_wall,
            pos=position_wall, rot=rotation_wall
        )

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(
        arena, category,
        pos=(np.random.randint(5, 35), 0,
             np.random.randint(position_door_axis_2 + 2,
                               35)) if rotation_wall == 90
        else (np.random.randint(position_door_axis_2 + 2, 35), 0,
              np.random.randint(5, 35)))

    rotation_agent = random_rotation() if  is_train else None
    position_agent = (
        np.random.randint(5, 35), 0, np.random.randint(
            5, position_door_axis_2 - 2)) if rotation_wall == 90 else (
        np.random.randint(5, position_door_axis_2 - 2), 0,
        np.random.randint(5, 35))

    arena = add_object(
        arena, 'Agent',
        pos=position_agent,
        rot=rotation_agent
    )

    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def push_test_1(arena, is_train=False):
    """
    The arena divided in 2 parts, the agent is required to push box
    to access the other side.
    """

    position_door_axis_1 = np.random.randint(2, 38)
    position_door_axis_2 = np.random.randint(10, 30)
    height_walls = random.choice([0.5, 2, 5, 10])

    size_wall_1 = (0.5, height_walls, - 1.15 + position_door_axis_1)
    size_wall_2 = (0.5, height_walls, 38.75 - position_door_axis_1)
    sizes_wall = [size_wall_1, size_wall_2]

    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])
    rotation_wall = random.choice([90, 0])

    if rotation_wall == 90:
        positions_wall = [(size_wall_1[-1] / 2, 0, position_door_axis_2),
                          (40 - size_wall_2[-1] / 2, 0, position_door_axis_2)]
    else:
        positions_wall = [(position_door_axis_2, 0, size_wall_1[-1] / 2),
                          (position_door_axis_2, 0, 40 - size_wall_2[-1] / 2)]

    for size_wall, position_wall in zip(sizes_wall, positions_wall):
        arena = add_object(
            arena, category, size=size_wall,
            pos=position_wall, rot=rotation_wall
        )

    category = random.choice(['Cardbox1', 'Cardbox2'])

    if rotation_wall == 90:
        arena = add_object(
            arena, category,
            size=(2, np.random.randint(2, 6), 2),
            pos=(position_door_axis_1, 0, position_door_axis_2))
    else:
        arena = add_object(
            arena, category,
            size=(2, np.random.randint(2, 6), 2),
            pos=(position_door_axis_2, 0, position_door_axis_1))

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(
        arena, category,
        pos=(np.random.randint(5, 35), 0,
             np.random.randint(position_door_axis_2 + 2,
                               35)) if rotation_wall == 90
        else (np.random.randint(position_door_axis_2 + 2, 35), 0,
              np.random.randint(5, 35)))

    rotation_agent = random_rotation() if is_train else None
    position_agent = (
        np.random.randint(5, 35), 0, np.random.randint(
            5, position_door_axis_2 - 2)) if rotation_wall == 90 else (
        np.random.randint(5, position_door_axis_2 - 2), 0,
        np.random.randint(5, 35))

    arena = add_object(
        arena, 'Agent',
        pos=position_agent,
        rot=rotation_agent
    )

    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def push_test_2(arena, is_train=False):
    """
    The arena contains a zone only accessible after pushing a box.
    """

    position_door_axis_1 = np.random.randint(8, 28)
    position_door_axis_2 = np.random.randint(8, 28)

    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])
    size_wall = (0.5, random.choice([0.5, 2, 5, 10]), 10)
    rotation_wall = random.choice([0, 90])

    if rotation_wall == 0:
        positions_wall = [
            (position_door_axis_1 - 1.5, 0, position_door_axis_2),
            (position_door_axis_1 + 1.5, 0, position_door_axis_2)]
    else:
        positions_wall = [
            (position_door_axis_1, 0, position_door_axis_2 - 1.5),
            (position_door_axis_1, 0, position_door_axis_2 + 1.5)]

    for position_wall in positions_wall:
        arena = add_object(
            arena, category, size=size_wall,
            pos=position_wall, rot=rotation_wall
        )

    category = random.choice(['Cardbox1', 'Cardbox2'])
    if rotation_wall == 0:
        arena = add_object(arena, category, size=(2, 2, 2), pos=(
            position_door_axis_1, 0, position_door_axis_2 - 6))
        arena = add_object(arena, category, size=(2, 2, 2), pos=(
            position_door_axis_1, 0, position_door_axis_2 + 6))
    else:
        arena = add_object(arena, category, size=(2, 2, 2), pos=(
            position_door_axis_1 - 6, 0, position_door_axis_2))
        arena = add_object(arena, category, size=(2, 2, 2), pos=(
            position_door_axis_1 + 6, 0, position_door_axis_2))

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(arena, category, size=(2, 2, 2),
                       pos=(position_door_axis_1, 0, position_door_axis_2))
    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    position_agent = random_pos() if  is_train else None
    rotation_agent = random_rotation() if  is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def tunnel_test_1(arena, is_train=False):
    """
    The arena divided in 2 parts, the agent is required to pass through a
    tunnel to access the other side.
    """

    position_door_axis_1 = np.random.randint(5, 35)
    position_door_axis_2 = np.random.randint(10, 30)

    height_walls = random.choice([0.5, 2, 5, 10])

    size_wall_1 = (0.5, height_walls, - 1.5 + position_door_axis_1)
    size_wall_2 = (0.5, height_walls, 38.5 - position_door_axis_1)
    sizes_wall = [size_wall_1, size_wall_2]

    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])
    rotation_wall = random.choice([90, 0])

    if rotation_wall == 90:
        positions_wall = [(size_wall_1[-1] / 2, 0, position_door_axis_2),
                          (40 - size_wall_2[-1] / 2, 0, position_door_axis_2)]
    else:
        positions_wall = [(position_door_axis_2, 0, size_wall_1[-1] / 2),
                          (position_door_axis_2, 0, 40 - size_wall_2[-1] / 2)]

    for size_wall, position_wall in zip(sizes_wall, positions_wall):
        arena = add_object(
            arena, category, size=size_wall,
            pos=position_wall, rot=rotation_wall
        )

    category = random.choice(['CylinderTunnel', 'CylinderTunnelTransparent'])
    if rotation_wall == 90:
        arena = add_object(
            arena, category, size=(2, np.random.randint(4, 6), 2),
            pos=(position_door_axis_1, 0, position_door_axis_2),
            rot=rotation_wall + 90)
    else:
        arena = add_object(
            arena, category, size=(2, np.random.randint(4, 6), 2),
            pos=(position_door_axis_2, 0, position_door_axis_1),
            rot=rotation_wall + 90)

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(
        arena, category,
        pos=(np.random.randint(5, 35), 0,
             np.random.randint(position_door_axis_2 + 2,
                               35)) if rotation_wall == 90
        else (np.random.randint(position_door_axis_2 + 2, 35), 0,
              np.random.randint(5, 35)))

    rotation_agent = random_rotation() if is_train else None
    position_agent = (
        np.random.randint(5, 35), 0, np.random.randint(
            5, position_door_axis_2 - 2)) if rotation_wall == 90 else (
        np.random.randint(5, position_door_axis_2 - 2), 0,
        np.random.randint(5, 35))

    arena = add_object(
        arena, 'Agent',
        pos=position_agent,
        rot=rotation_agent
    )

    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def tunnel_test_2(arena, is_train=False):
    """ Reward is walled, access through a tunnel. """

    position_x = np.random.randint(10, 30)
    position_y = np.random.randint(10, 30)
    height_walls = random.choice([0.5, 2, 5, 10])
    size_walls = (0.5, height_walls, 8)
    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])

    inv = random.choice([1, -1])

    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x + inv * 4, 0, position_y), rot=0)
    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x, 0, position_y + inv * 4.25), rot=90)
    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x - inv * 4, 0, position_y), rot=0)
    arena = add_object(arena, category, size=(0.5, height_walls, 3),
                       pos=(
                           position_x + inv * 2.75, 0,
                           position_y - inv * 4.25),
                       rot=90)
    arena = add_object(arena, category, size=(0.5, height_walls, 3),
                       pos=(
                           position_x - inv * 2.75, 0,
                           position_y - inv * 4.25),
                       rot=90)

    category = random.choice(['CylinderTunnel', 'CylinderTunnelTransparent'])
    arena = add_object(arena, category, size=(2, random.choice([4, 5, 6]), 2),
                       pos=(position_x, 0, position_y - inv * 4.25), rot=0)

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(arena, category, pos=(position_x, 0, position_y))
    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    position_agent = random_pos() if is_train else None
    rotation_agent = random_rotation() if is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def ramp_test_1(arena, is_train=False):
    """ Reward is walled, access through a ramp. """

    position_x = np.random.randint(15, 25)
    position_y = np.random.randint(15, 25)
    height_walls = random.choice([0.5, 1, 2, 3, 4, 5, 6])
    size_walls = (0.5, height_walls, 8)
    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])
    inv = random.choice([1, -1])

    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x + inv * 4, 0, position_y), rot=0)
    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x - inv * 4, 0, position_y), rot=0)
    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x, 0, position_y + inv * 4.25), rot=90)
    arena = add_object(arena, category, size=size_walls,
                       pos=(position_x, 0, position_y - inv * 4.25), rot=90)

    category = random.choice(['Ramp'])
    arena = add_object(arena, category, size=(8, height_walls, 8),
                       pos=(position_x + inv * 8.5, 0, position_y),
                       rot=90 if inv == 1 else 270)

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(arena, category, pos=(position_x, 0, position_y))
    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    position_agent = random_pos() if is_train else None
    rotation_agent = random_rotation() if is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def ramp_test_2(arena, is_train=False):
    """
    The arena is divided in 2 parts, the agent is required to climb up a ramp
    to access the other side.
    """
    position_ramp_1 = np.random.randint(5, 15)
    position_ramp_2 = np.random.randint(20, 35)
    height_walls = random.choice([0.5, 1, 2, 3, 4, 5])
    size_wall = (2, height_walls, 40)
    category = random.choice(['Wall', 'WallTransparent', 'DeathZone'])

    if random.random() > 0.5:
        rotation_wall = 0
        arena = add_object(arena, category, size=size_wall, pos=(20, 0, 20),
                           rot=rotation_wall)
        category = random.choice(['Ramp'])
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(20 + 5, 0, position_ramp_1), rot=90)
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(20 - 5, 0, position_ramp_2), rot=270)
    else:
        rotation_wall = 90
        arena = add_object(arena, category, size=size_wall, pos=(20, 0, 20),
                           rot=rotation_wall)
        category = random.choice(['Ramp'])
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(position_ramp_1, 0, 20 - 5), rot=180)
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(position_ramp_2 - 5, 0, 20 + 5), rot=0)

    position_agent = (
        np.random.randint(5, 38), 0,
        np.random.randint(5, 20 - 10)) if rotation_wall == 90 else (
        (np.random.randint(5, 20 - 10), 0, np.random.randint(5, 38))
    )

    rotation_agent = random_rotation() if  is_train else None
    arena = add_object(
        arena, 'Agent',
        pos=position_agent,
        rot=rotation_agent,
    )

    category = random.choice(['GoodGoal'])
    arena = add_object(arena, category,
                       pos=(np.random.randint(5, 38), 0,
                            np.random.randint(20 + 10, 38))
                       if rotation_wall == 90
                       else (np.random.randint(20 + 10, 38), 0,
                             np.random.randint(5, 38)))

    for _ in range(2):
        category = random.choice(['GoodGoalMulti'])
        arena = add_object(arena, category)

    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def ramp_test_3(arena, is_train=False):
    """ Reward in a platform, access through a ramp. """

    position_x = np.random.randint(15, 25)
    position_y = np.random.randint(15, 25)
    height_platform = random.choice([0.5, 1, 2, 3, 4, 5, 6])
    inv = random.choice([1, -1])
    category = random.choice(
        ['Wall', 'WallTransparent', 'Cardbox1', 'Cardbox2'])
    arena = add_object(arena, category, size=(4, height_platform, 4),
                       pos=(position_x, 0, position_y), rot=90)
    arena = add_object(arena, 'Ramp', size=(8, height_platform, 8),
                       pos=(position_x - inv * 6, 0, position_y),
                       rot=270 if inv == 1 else 90)
    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(arena, category,
                       pos=(position_x, height_platform + 0.25, position_y))
    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    position_agent = random_pos() if is_train else None
    rotation_agent = random_rotation() if is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def reasoning_step_1(arena, is_train=False):
    """ Move box to correct location to access food. """

    position_x = np.random.randint(15, 25)
    position_y = np.random.randint(15, 25)
    height_platform = random.choice([2, 3, 4])
    inv = random.choice([1, -1])
    category = random.choice(['Wall', 'WallTransparent'])
    arena = add_object(arena, category, size=(4, height_platform, 4),
                       pos=(position_x + inv * 8.5, 0, position_y), rot=90)
    arena = add_object(arena, category, size=(4, height_platform, 4),
                       pos=(position_x, 0, position_y), rot=90)
    category = random.choice(['Cardbox1', 'Cardbox2'])
    arena = add_object(
        arena, category, size=(4, height_platform, 4),
        pos=(position_x + inv * 4.25, 0, position_y + inv * 6), rot=90)
    arena = add_object(arena, 'GoodGoalMulti', pos=(
        position_x, height_platform + 0.25, position_y))

    arena = add_object(arena, 'Ramp', size=(8, height_platform, 8),
                       pos=(position_x - inv * 6, 0, position_y),
                       rot=270 if inv == 1 else 90)

    arena = add_object(arena, 'GoodGoal', pos=(
        position_x + inv * 8.5, height_platform + 0.25, position_y))
    #category = random.choice(['BadGoal'])
    #arena = add_object(arena, category)

    position_agent = random_pos(with_respect_to_center='far')   if is_train else None
    rotation_agent = random_rotation() if is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def reasoning_step_2(arena, is_train=False):
    """ Move box to correct location to access food. """

    position_x = np.random.randint(15, 25)
    position_y = np.random.randint(15, 25)
    height_platform = random.choice([2, 3, 4])
    inv = random.choice([1, -1])
    category = random.choice(['Wall', 'WallTransparent'])
    arena = add_object(arena, category, size=(4, height_platform - 1, 4),
                       pos=(position_x + inv * 8.5, 0, position_y), rot=90)
    arena = add_object(arena, category, size=(4, height_platform - 1, 4),
                       pos=(position_x, 0, position_y), rot=90)
    category = random.choice(['Cardbox1', 'Cardbox2'])
    arena = add_object(arena, category, pos=(
        position_x - inv * 1.5, height_platform + 0.25, position_y),
                       size=(0.5, 8, 4), rot=0)
    arena = add_object(arena, 'Ramp', size=(8, height_platform, 8),
                       pos=(position_x - inv * 6, 0, position_y),
                       rot=270 if inv == 1 else 90)

    arena = add_object(arena, 'GoodGoal', pos=(
        position_x + inv * 8.5, height_platform + 0.25, position_y))
    #category = random.choice(['BadGoal'])
    #arena = add_object(arena, category)

    position_agent = random_pos(with_respect_to_center='far')  if is_train else None
    rotation_agent = random_rotation() if is_train else None
    
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def reasoning_step_3(arena, is_train=False):
    """ Move u-shaped object to correct location to access food. """

    position_x = np.random.randint(15, 25)
    position_y = np.random.randint(15, 25)
    size_wall = (4, 0, 40)
    category = random.choice(['DeathZone'])

    if random.random() > 0.5:
        rotation_wall = 0
        arena = add_object(arena, category, size=size_wall,
                           pos=(position_x, 0, 20), rot=rotation_wall)
        category = random.choice(['UObject'])
        arena = add_object(arena, category, size=(8, 0.3, 8),
                           pos=(20 - 6, 0, 20 - 6), rot=rotation_wall)
    else:
        rotation_wall = 90
        arena = add_object(arena, category, size=size_wall,
                           pos=(20, 0, position_y), rot=rotation_wall)
        category = random.choice(['UObject'])
        arena = add_object(arena, category, size=(8, 0.3, 8),
                           pos=(20 - 6, 0, 20 - 6), rot=rotation_wall)

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])

    arena = add_object(
        arena, category,
        pos=(np.random.randint(5, 35), 0,
             np.random.randint(position_y + 10, 35)) if rotation_wall == 90
        else (np.random.randint(position_x + 10, 35), 0,
              np.random.randint(5, 35)))

    rotation_agent = random_rotation() if is_train else None
    position_agent = (np.random.randint(5, 35), 0,
                      np.random.randint(
                          position_y + 10, 35)) if rotation_wall == 90 else (
        np.random.randint(position_x + 10, 35),
        0, np.random.randint(5, 35))

    arena = add_object(arena, 'Agent', pos=position_agent, rot=rotation_agent)

    #category = random.choice(['BadGoal'])
    #arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def narrow_spaces_1(arena, is_train=False):
    """ Navigate in restricted spaces. """

    height_walls = random.choice([0.5, 1, 2, 3, 4, 5])
    size_wall = (4, height_walls, 40)
    category = random.choice(['Wall', 'WallTransparent'])

    rotation_wall = 0
    arena = add_object(arena, category, size=size_wall, pos=(20, 0, 20),
                       rot=rotation_wall)

    done = False
    while done is False:
        possible_positions_reward = []
        for i in range(4):
            if random.random() > 0.5:
                pos = (9, 0, i * (40 // 4) + 2)
                possible_positions_reward.append((2, height_walls + 0.25, pos[2]))
                arena = add_object(arena, category,
                                   size=(18, height_walls, 4),
                                   pos=pos,
                                   rot=rotation_wall)

        for i in range(4):
            if random.random() > 0.5:
                pos = (40 - 9, 0, i * (40 // 4) + 2)
                possible_positions_reward.append((38, height_walls + 0.25, pos[2]))
                arena = add_object(arena, category,
                                   size=(18, height_walls, 4),
                                   pos=pos,
                                   rot=rotation_wall)

        if len(possible_positions_reward) > 1:
            done = True

    arena = add_object(
        arena, "DeathZone", size=(40, 0, 40), pos=(20, 0, 20),
        rot=rotation_wall)

    rotation_agent = random_rotation() if is_train else None
    position_agent = random.choice(possible_positions_reward)
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    pos_reward = random.choice(possible_positions_reward)
    while pos_reward == position_agent:
        pos_reward = random.choice(possible_positions_reward)

    arena = add_object(arena, random.choice(["GoodGoal", "GoodGoalMulti"]),
                       pos=pos_reward)

    return arena, position_agent, rotation_agent


def narrow_spaces_2(arena, is_train=False):
    """ Navigate in restricted spaces. """

    position_ramp_1 = np.random.randint(5, 15)
    position_ramp_2 = np.random.randint(20, 35)
    height_walls = random.choice([0.5, 1, 2, 3, 4, 5])
    wall_width = random.choice([2, 3, 4, 5, 6])
    size_wall = (wall_width, height_walls, 40)
    category = random.choice(['Wall', 'WallTransparent'])

    if random.random() > 0.5:
        rotation_wall = 0
        arena = add_object(arena, category, size=size_wall, pos=(20, 0, 20),
                           rot=rotation_wall)
        category = random.choice(['Ramp'])
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(20 + 4 + wall_width / 2, 0, position_ramp_1),
                           rot=90)
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(20 - 4 - wall_width / 2, 0, position_ramp_2),
                           rot=270)
        arena = add_object(arena, "GoodGoal", pos=(
            20, height_walls + 0.25, np.random.randint(5, 35)),
                           rot=rotation_wall)
    else:
        rotation_wall = 90
        arena = add_object(arena, category, size=size_wall, pos=(20, 0, 20),
                           rot=rotation_wall)
        category = random.choice(['Ramp'])
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(position_ramp_1, 0, 20 - 4 - wall_width / 2),
                           rot=180)
        arena = add_object(arena, category, size=(8, height_walls, 8),
                           pos=(
                               position_ramp_2 - 5, 0,
                               20 + 4 + wall_width / 2),
                           rot=0)
        arena = add_object(arena, "GoodGoal", pos=(
            np.random.randint(5, 35), height_walls + 0.25, 20),
                           rot=rotation_wall)

    arena = add_object(arena, "HotZone", size=(40, 0, 40), pos=(20, 0, 20),
                       rot=rotation_wall)

    position_agent = random_pos() if is_train else None
    rotation_agent = random_rotation() if  is_train else None
    arena = add_object(arena, "Agent", pos=position_agent, rot=rotation_agent)

    return arena, position_agent, rotation_agent


def preference_test_1(arena, is_train=False):
    """
    The arena divided in 2 parts, the agent is required to push box
    to access the other side.
    """

    height_walls = 5

    posU_1 = (np.random.randint(21, 30), 0, np.random.randint(10, 30))
    posU_2 = (np.random.randint(10, 19), 0, np.random.randint(10, 30))

    long_wall_siz = (1, height_walls, 10)

    short_wall_siz = (5, height_walls, 1)

    pos_short_1 = (posU_1[0] + 0.5 + 5 / 2, 0, posU_1[2] + 5)
    pos_short_2 = (posU_1[0] + 0.5 + 5 / 2, 0, posU_1[2] - 5)

    arena = add_object(arena, 'Wall', size=long_wall_siz, pos=posU_1, rot=0)
    arena = add_object(arena, 'Wall', size=short_wall_siz, pos=pos_short_1,
                       rot=0)
    arena = add_object(arena, 'Wall', size=short_wall_siz, pos=pos_short_2,
                       rot=0)

    pos_short_1 = (posU_2[0] - 0.5 - 5 / 2, 0, posU_2[2] + 5)
    pos_short_2 = (posU_2[0] - 0.5 - 5 / 2, 0, posU_2[2] - 5)

    arena = add_object(arena, 'Wall', size=long_wall_siz, pos=posU_2, rot=0)
    arena = add_object(arena, 'Wall', size=short_wall_siz, pos=pos_short_1,
                       rot=0)
    arena = add_object(arena, 'Wall', size=short_wall_siz, pos=pos_short_2,
                       rot=0)

    position_agent = (20, 0, 20)
    rotation_agent = random_rotation() if is_train else None

    arena = add_object(arena, 'Agent', pos=position_agent, rot=rotation_agent)
    rew_size = np.random.randint(1, 5)
    arena = add_object(arena, 'GoodGoal', size=(rew_size, rew_size, rew_size),
                       pos=(posU_1[0] + 3, posU_1[1], posU_1[2]))
    rew_size_2 = np.random.randint(1, 5)
    arena = add_object(arena, 'GoodGoal',
                       size=(rew_size_2, rew_size_2, rew_size_2),
                       pos=(posU_2[0] - 3, posU_2[1], posU_2[2]))

    return arena, position_agent, rotation_agent


def blackout_test_1(arena, is_train=False):
    """
    The arena divided in 2 parts, the agent is required to push box
    to access the other side.
    """

    position_door_axis_1 = np.random.randint(2, 38)
    position_door_axis_2 = np.random.randint(10, 30)
    height_walls = random.choice([0.5, 2, 5, 10])

    size_wall_1 = (0.5, height_walls, - 1.15 + position_door_axis_1)
    size_wall_2 = (0.5, height_walls, 38.75 - position_door_axis_1)
    sizes_wall = [size_wall_1, size_wall_2]

    category = random.choice(['WallTransparent'])
    rotation_wall = random.choice([90, 0])

    if rotation_wall == 90:
        positions_wall = [(size_wall_1[-1] / 2, 0, position_door_axis_2),
                          (40 - size_wall_2[-1] / 2, 0, position_door_axis_2)]
    else:
        positions_wall = [(position_door_axis_2, 0, size_wall_1[-1] / 2),
                          (position_door_axis_2, 0, 40 - size_wall_2[-1] / 2)]

    for size_wall, position_wall in zip(sizes_wall, positions_wall):
        arena = add_object(
            arena, category, size=size_wall,
            pos=position_wall, rot=rotation_wall
        )

    category = random.choice(['GoodGoal', 'GoodGoalMulti'])
    arena = add_object(
        arena, category,
        pos=(np.random.randint(5, 35), 0,
             np.random.randint(position_door_axis_2 + 2,
                               35)) if rotation_wall == 90
        else (np.random.randint(position_door_axis_2 + 2, 35), 0,
              np.random.randint(5, 35)))

    position_agent = (20, 0, 2) if rotation_wall == 90 else (2, 0, 20)
    rotation_agent = 0 if rotation_wall == 90 else 90
    arena = add_object(
        arena, 'Agent',
        pos=position_agent,
        rot=rotation_agent)

    category = random.choice(['BadGoal'])
    arena = add_object(arena, category)

    return arena, position_agent, rotation_agent


def create_wall(A, B, z_size, obj='CylinderTunnel', gap=2):
    # A is the statrting point, B is  the endpoint
    # dor can be empty, cylinder, ramp, box

    Ax, Ay = A
    Bx, By = B

    if Bx == Ax:
        if Ay != 0 and Ay != 40:
            Ay += 0.5
        if By != 0 and By != 40:
            By -= 0.5

        x_size = 1
        y_size = round((By - Ay), 2)

        y_pos = round(Ay + (By - Ay) / 2, 2)
        x_pos = Bx

        if obj == 'door':
            y_size_1 = (y_size - gap) / 2
            y_size_2 = (y_size - gap) / 2

            y_pos_1 = y_pos - gap / 2 - (y_size - gap) / 4
            y_pos_2 = y_pos + gap / 2 + (y_size - gap) / 4

            return ((x_size, z_size, y_size_1), (x_pos, 0.0, y_pos_1),
                    (x_size, z_size, y_size_2), (x_pos, 0.0, y_pos_2))

        if obj == 'CylinderTunnel':
            return (gap, gap, gap), (x_pos, 0.0, y_pos)
        if obj == 'Cardbox2':
            gap_box = gap - 0.2
            return (gap_box, gap_box, gap_box), (x_pos, 0.0, y_pos)
        if obj == 'Cardbox1':
            gap_box = gap - 0.2
            return (gap_box, gap_box, gap_box), (x_pos, 0.0, y_pos)
        if obj == 'Ramp':
            return (gap, 2, 3), (x_pos, 0.0, y_pos)

    if By == Ay:

        if Ax != 0 and Ax != 40:
            Ax += 0.5
        if Bx != 0 and Bx != 40:
            Bx -= 0.5

        y_size = 1
        x_size = round((Bx - Ax), 2)

        x_pos = round(Ax + (Bx - Ax) / 2, 2)
        y_pos = By

        if obj == 'door':
            x_size_1 = (x_size - gap) / 2
            x_size_2 = (x_size - gap) / 2

            x_pos_1 = x_pos - gap / 2 - (x_size - gap) / 4
            x_pos_2 = x_pos + gap / 2 + (x_size - gap) / 4

            return ((x_size_1, z_size, y_size), (x_pos_1, 0.0, y_pos),
                    (x_size_2, z_size, y_size), (x_pos_2, 0.0, y_pos))

        if obj == 'CylinderTunnel':
            return (gap, gap, gap), (x_pos, 0.0, y_pos)
        if obj == 'Cardbox2':
            gap_box = gap - 0.2
            return (gap_box, gap_box, gap_box), (x_pos, 0.0, y_pos)
        if obj == 'Cardbox1':
            gap_box = gap - 0.2
            return (gap_box, gap_box, gap_box), (x_pos, 0.0, y_pos)
        if obj == 'Ramp':
            return (gap, 2, 3), (x_pos, 0.0, y_pos)

    return (x_size, z_size, y_size), (x_pos, 0.0, y_pos)


if __name__ == '__main__':

    import os

    target_dir = "../../"
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for i in range(100):
        arena = cross_test("")
        save_name = '{}/{}'.format(target_dir, "cross_test_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = push_test_1("")
        save_name = '{}/{}'.format(target_dir, "push_test_1_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = push_test_2("")
        save_name = '{}/{}'.format(target_dir, "push_test_2_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = tunnel_test_1("")
        save_name = '{}/{}'.format(target_dir, "tunnel_test_1_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = tunnel_test_2("")
        save_name = '{}/{}'.format(target_dir, "tunnel_test_2_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = ramp_test_1("")
        save_name = '{}/{}'.format(target_dir, "ramp_test_1_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = ramp_test_2("")
        save_name = '{}/{}'.format(target_dir, "ramp_test_2_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)

    for i in range(100):
        arena = ramp_test_3("")
        save_name = '{}/{}'.format(target_dir, "ramp_test_3_num{}".format(i))
        write_arena(save_name, random.choice([500, 1000]), arena)
