import random
import pickle
import os.path
import pandas as pd
import numpy as np

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

reward_objects = ['GoodGoal','GoodGoalBounce','BadGoal','BadGoalBounce','GoodGoalMulti','GoodGoalMultiBounce']
immovable_objects = ['Wall','WallTransparent','Ramp','CylinderTunnel','CylinderTunnelTransparent']
movable_objects=['Cardbox1','Cardbox2','UObject','LObject','LObject2']
zone_objects = ['DeathZone','HotZone']

#time_limits = [250,500,1000]
time_limits = [250,500,1000]

def add_object(s, object_name, pos=None, size=None, RGB=None):
    s += "    - !Item \n      name: {} \n".format(object_name)

    if RGB is None:
        if object_name is 'Ramp':
            RGB=(255,0,255)
        elif object_name is 'Tunnel':
            RGB=(153,153,153) 
        elif object_name is 'Wall':
            RGB=(153,153,153) 

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

def binomial_learning(narenas):
    for a in range(narenas):
        arena = ''    

        #immovable
        num_immovable = np.random.randint(0,3)
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
        if num_immovable+num_movable<2:  #avoid to many objects
            num_zones = np.random.randint(0,2)
            nums = np.random.binomial(num_zones, [0.5,0.5] )
            for i,n in enumerate(nums):
                for _ in range(n):
                    arena = add_object(arena,zone_objects[i],size=random_size_zone())

        #Rewards
        num_rewards = np.random.randint(1,10)
        nums = np.random.binomial(num_rewards, [0.01,0.05,0.5,0.1,0.15,0.1] )
        if  nums[[0,1,4,5]].sum()==0: 
            nums[0]=1
        for i,n in enumerate(nums):
            for _ in range(n):
                arena = add_object(arena,reward_objects[i],size=random_size_reward())    
            
        write_arena(str(a),[random.choice(time_limits)], arena)

def cv_learning():
    #Learn green ball
    arena = add_object('','GoodGoal',size=random_size_reward())
    write_arena('C1_good1',time_limits, arena)

    #Learn green ball bouncing
    arena = add_object('','GoodGoalBounce',size=random_size_reward())
    write_arena('C1_goodbounce1',time_limits, arena)

    #Learn gold ball behavior
    arena = add_object('','GoodMultiGoal',size=random_size_reward())
    arena = add_object(arena,'GoodMultiGoal',size=random_size_reward())
    write_arena('C1_goodmulti2',time_limits, arena)

    #Learn gold ball behavior bouncing
    arena = add_object('','GoodMultiGoalBounce',size=random_size_reward())
    arena = add_object(arena,'GoodMultiGoalBounce',size=random_size_reward())
    write_arena('C1_goodmultibounce2',time_limits, arena)

    #Learn size matters
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoal',size=random_size_reward())
    write_arena('C1_good2', time_limits, arena)

    #Learn size matters bouncing
    arena = add_object('','GoodGoalBounce',size=random_size_reward())
    arena = add_object(arena,'GoodGoalBounce',size=random_size_reward())
    write_arena('C1_goodbounce2', time_limits, arena)

    #Learn trade-off easy-more if moving is larger
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoalBounce',size=random_size_reward())
    write_arena('C1_good_goodbounce', time_limits, arena)

    #Learn to prefer food that keeps you alive first
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoalMulti',size=random_size_reward())
    write_arena('C1_good_goodmulti', time_limits, arena)

    #Learn to prefer food that keeps you alive first, even if harder to get
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoalMultiBounce',size=random_size_reward())
    write_arena('C1_good_goodmultibounce', time_limits, arena)

    #Learn to die early if nothing there
    arena = add_object('','BadGoal',size=random_size_reward())
    arena = add_object('','BadGoal',size=random_size_reward())
    write_arena('C1_bad1',time_limits, arena)

    #Learn to die early if nothing there
    arena = add_object('','BadGoalBounce',size=random_size_reward())
    write_arena('C1_badbounce1',time_limits, arena)

    #Learn to avoid red balls if there is food
    arena = add_object('','BadGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoal',size=random_size_reward())
    write_arena('C1_badgood2',time_limits, arena)

    #Learn to avoid red balls if there is food
    arena = add_object('','BadGoal',size=random_size_reward())
    arena = add_object(arena,'GoodGoalMulti',size=random_size_reward())
    write_arena('C1_badgoodmulti2',time_limits, arena)

    #Learn to navigate around many bad
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'BadGoal',size=random_size_reward())
    arena = add_object(arena,'BadGoal',size=random_size_reward())
    arena = add_object(arena,'BadGoal',size=random_size_reward())
    arena = add_object(arena,'BadGoal',size=random_size_reward())
    write_arena('C1_manybad',time_limits, arena)

    #Learn to navigate around many bad
    arena = add_object('','GoodGoal',size=random_size_reward())
    arena = add_object(arena,'BadGoalBounce',size=random_size_reward())
    arena = add_object(arena,'BadGoalBounce',size=random_size_reward())
    arena = add_object(arena,'BadGoalBounce',size=random_size_reward())
    arena = add_object(arena,'BadGoalBounce',size=random_size_reward())
    write_arena('C1_manybadbounce',time_limits, arena)



binomial_learning(5000)
#cv_learning()
