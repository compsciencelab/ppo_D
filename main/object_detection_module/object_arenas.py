import random
from .settings import objects, labels
from animal.arenas.utils.edit_arenas import add_object, write_arena


def create_object_arena(target_path, arena_name, num_objects=7, time=250):
    """ Empty arena with ´num_objects´ objects of the same type"""

    arena = ''
    object = random.choice(objects)

    for _ in range(num_objects):
        arena = add_object(arena, object)

    save_name = '{}/{}'.format(target_path, arena_name)
    write_arena(save_name, time, arena, blackouts=None)

    return object, labels[object]
