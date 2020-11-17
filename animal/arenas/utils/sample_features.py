import math
import random


ranges = {
    'close': [0, 5.0],
    'medium': [5.0, 10.0],
    'far': [10.0, 60.0]
}


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[2] - p1[2])**2)


def random_pos(with_respect_to_center=None):

    if with_respect_to_center in ['close', 'medium', 'far']:
        return sample_position_with_respect_to((20., 0., 20.), range='close')

    return (random.randint(10, 390) / 10., 0., random.randint(10, 390) / 10.)


def random_rotation():
    return random.randint(0, 360)


def random_size(category):

    if category in ['GoodGoal', 'GoodGoalBounce', 'BadGoal', 'BadGoalBounce', 'GoodGoalMulti', 'GoodGoalMultiBounce']:
        #according to docs it's 0.5-5
        s = random.randint(5, 50)/10
        s = (s,s,s)
    elif category in ['Wall', 'WallTransparent']:
        s = (random.randint(20, 250) / 10., random.randint(20, 70) / 10., random.randint(20, 250) / 10.)
    elif category in ['Ramp']:
        s = (random.randint(5, 400) / 10., random.randint(1, 200) / 10., random.randint(5, 400) / 10.)
    elif category in ['CylinderTunnel', 'CylinderTunnelTransparent']:
        s = (random.randint(25, 100) / 10., random.randint(25, 100) / 10., random.randint(25, 100) / 10.)
    elif category in ['Cardbox1', 'Cardbox2']:
        s = (random.randint(5, 100) / 10., random.randint(5, 100) / 10., random.randint(5, 100) / 10.)
    elif category in ['UObject', 'LObject', 'LObject2']:
        s = (random.randint(10, 50) / 10., random.randint(3, 20) / 10., random.randint(30, 200) / 10.)
    elif category in ['DeathZone', 'HotZone']:
        s = (random.randint(10, 100) / 10., 0., random.randint(10, 100) / 10.)

    return s


def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def sample_position_with_respect_to(reference_position, range='far'):
    """
    Sample random position within the specified range from reference_position.
        - close: dist in range [0, 7.5)
        - medium: dist in range [7.5, 15)
        - far: dist in range [15, +)
    """

    position = random_pos()

    lower_bound = ranges[range][0]
    upper_bound = ranges[range][1]

    while not lower_bound <= distance(reference_position, position) < upper_bound:
        position = random_pos()

    return position
