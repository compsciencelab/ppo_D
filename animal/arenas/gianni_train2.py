import random
import pickle
import os.path
import pandas as pd
import numpy as np


#Importat info from documentation 
#Note that all tests are pass/fail based on achieving a score above a certain threshold. 
#In most cases, this corresponds to retrieving the only piece of food in the environment. 


def random_pos():
    return (random.randint(1, 400)/10, random.randint(1, 400)/10, random.randint(1, 400)/10)

def random_size_reward():
    #according to docs it's 0.5-5
    s = random.randint(5, 50)/10
    return (s,s,s)

def random_size_zone():
    #according to docs it's 1-40
    s1 = random.randint(1,40)
    s2 = random.randint(1,40)
    return (s1,0,s2)

pos_reward_objects = ['GoodGoal','GoodGoalBounce','GoodGoalMulti','GoodGoalMultiBounce']
immovable_objects = ['Wall','WallTransparent','Ramp','CylinderTunnel','CylinderTunnelTransparent']
movable_objects=['Cardbox1','Cardbox2','UObject','LObject','LObject2']
zone_objects = ['DeathZone','HotZone']
neg_reward_objects = ['BadGoal','BadGoalBounce']

all_objects = pos_reward_objects+immovable_objects+movable_objects+zone_objects+neg_reward_objects
barrier_objects = immovable_objects+movable_objects+zone_objects
reward_objects = pos_reward_objects+neg_reward_objects

time_limits = [250,500,1000]

def add_object(s, object_name, pos=None, size=None, RGB=None, set_reward_size = False):
    s += "    - !Item \n      name: {} \n".format(object_name)

    if RGB is None:
        if object_name is 'Ramp':
            RGB=(255,0,255)
        elif object_name is 'Tunnel':
            RGB=(153,153,153) 
        elif object_name is 'Wall':
            RGB=(153,153,153) 

    if size is None and set_reward_size and object_name in reward_objects:
        size = random_size_reward()

    if pos is not None:
        s+= "      positions: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(pos[0],pos[1],pos[2])
    if size is not None:
        s+= "      sizes: \n      - !Vector3 {{x: {}, y: {}, z: {}}}\n".format(size[0],size[1],size[2])
    if RGB is not None:
        s+= "      colors: \n      - !RGB {{r: {}, g: {}, b: {}}}\n".format(RGB[0],RGB[1],RGB[2])
    if size is None and object_name in reward_objects:
        print("Warning: Reward without size")
    return s

def write_arena(fname, times, arena_str):
    for t in times:
        with open("{}_{}.yaml".format(fname,t), 'w+') as f:
            f.write("!ArenaConfig \narenas: \n  0: !Arena \n    t: {} \n    items:\n".format(t))
            f.write(arena_str)


random.seed(1)
np.random.seed(1)

def obj_1(repeats = 1):
# If I have only one object in the arena it has to be food (Hp)
# So I only include positive rewards here which can terminate the episode earlier
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            arena = ''
            arena = add_object(arena,a)
            write_arena('a1_i{}_r{}'.format(i,r),[random.choice(time_limits)], arena)


def obj_2(repeats = 1):
# If I have two objects in the arena it has to be food and anything else, including food again
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            for i2,a2 in enumerate(all_objects):
                arena = ''
                arena = add_object(arena,a)
                arena = add_object(arena,a2)
                write_arena('a2_i{}_i2{}_r{}'.format(i,i2,r),[random.choice(time_limits)], arena)

def obj_3(repeats = 1):
# If I have three objects in the arena it has to be one food,anything else and non_reward objects
# Because I am placing it randomly having two barrier objects helps in making the arena harder to navigate
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            for i2,a2 in enumerate(all_objects):
                for i3,a3 in enumerate(all_objects):
                    arena = ''
                    arena = add_object(arena,a)
                    arena = add_object(arena,a2)
                    arena = add_object(arena,a3)
                    write_arena('a3_i{}_i2{}_i3{}_r{}'.format(i,i2,i3,r),[random.choice(time_limits)], arena)
 

def obj_4(repeats = 1):
# If I have three objects in the arena it has to be one food,anything else and non_reward objects
# Because I am placing it randomly having two barrier objects helps in making the arena harder to navigate
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            for i2,a2 in enumerate(immovable_objects):
                for i3,a3 in enumerate(all_objects):
                    for i4,a4 in enumerate(all_objects):
                        arena = ''
                        arena = add_object(arena,a)
                        arena = add_object(arena,a2)
                        arena = add_object(arena,a3)
                        arena = add_object(arena,a4)
                        write_arena('a3_i{}_i2{}_i3{}_i4{}_r{}'.format(i,i2,i3,i4,r),[random.choice(time_limits)], arena)
 

def c2_preferences(repeats = 1):
# If I have three objects in the arena it has to be one food,anything else and non_reward objects
# Because I am placing it randomly having two barrier objects helps in making the arena harder to navigate
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(['GoodGoal','GoodGoalBounce']):
            for i2,a2 in enumerate(pos_reward_objects):
                for i3,a3 in enumerate(pos_reward_objects):
                    arena = ''
                    arena = add_object(arena,a)
                    arena = add_object(arena,a2)
                    arena = add_object(arena,a3)
                    write_arena('c2_i{}_i2{}_i3{}_r{}'.format(i,i2,i3,r),[random.choice(time_limits)], arena)

def c3_obstacles(repeats = 1):
# If I have three objects in the arena it has to be one food,anything else and non_reward objects
# Because I am placing it randomly having two barrier objects helps in making the arena harder to navigate
# Positive rewards have always a given random size
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            for i2,a2 in enumerate(immovable_objects):
                for i3,a3 in enumerate(all_objects):
                    arena = ''
                    arena = add_object(arena,a)
                    arena = add_object(arena,a2)
                    arena = add_object(arena,a3)
                    write_arena('c3_i{}_i2{}_i3{}_r{}'.format(i,i2,i3,r),[random.choice(time_limits)], arena)


def c3_see_throught(repeats = 1):
# If I have three objects in the arena it has to be one food,anything else and non_reward objects
# Because I am placing it randomly having two barrier objects helps in making the arena harder to navigate
# Positive rewards have always a given random size
    st_objects = ['WallTransparent','CylinderTunnelTransparent','DeathZone']
    for r in range(repeats):
        for i,a in enumerate(pos_reward_objects):
            for i2,a2 in enumerate(st_objects):
                for i3,a3 in enumerate(st_objects):
                    arena = ''
                    arena = add_object(arena,a)
                    arena = add_object(arena,a2)
                    arena = add_object(arena,a3)
                    write_arena('c3st_i{}_i2{}_i3{}_r{}'.format(i,i2,i3,r),[random.choice(time_limits)], arena)


def messy_arenas(narenas):
    for a in range(narenas):
        arena = ''    
        #immovable
        num_immovable = np.random.randint(1,3)
        nums = np.random.binomial(num_immovable, [0.5,0.2,0.1,0.1,0.1] )
        for i,n in enumerate(nums):
            for _ in range(n):
                arena = add_object(arena,immovable_objects[i])
        #movable
        num_movable = np.random.randint(0,2)
        nums = np.random.binomial(num_movable, [0.2,0.2,0.2,0.2,0.2] )
        for i,n in enumerate(nums):
            for _ in range(n):
                arena = add_object(arena,movable_objects[i])

        #Zones
        if num_immovable+num_movable<=2:  #avoid to many objects
            num_zones = np.random.randint(0,2)
            nums = np.random.binomial(num_zones, [0.5,0.5] )
            for i,n in enumerate(nums):
                for _ in range(n):
                    arena = add_object(arena,zone_objects[i])

        #Rewards
        num_rewards = np.random.randint(2,8)
        nums = np.random.binomial(num_rewards, [0.01,0.05,0.2,0.1,0.2,0.1] )
        if  nums[[0,1,4,5]].sum()==0: 
            nums[0]=1
        for i,n in enumerate(nums):
            for _ in range(n):
                arena = add_object(arena,reward_objects[i])    
            
        write_arena('m{}'.format(a),[random.choice(time_limits)], arena)

# def c2_preferences(repeats = 1):
# # This still contains just 3 objects but drawn from a different distribution (not all equal)
# # as second object is always a position reward
#     for r in range(repeats):
#         arena = ''
#         #I need at least one green to create a potential preferencial situation
#         arena = add_object(arena, random.choice(['GoodGoal','GoodGoalBounce']) )
#         arena = add_object(arena, random.choice(pos_reward_objects) )
#         arena = add_object(arena, random.choice(all_objects)) 
#         write_arena('c2_r{}'.format(r),[random.choice(time_limits)], arena)




if __name__ == "__main__":    
    obj_1()
    obj_2()
    obj_3()
    c2_preferences()
    c3_obstacles()
    c3_see_throught()
    messy_arenas(1000)
