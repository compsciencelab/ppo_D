import importlib.util
import glob
from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

def analyze_arena(arena):
    tot_reward = 0
    max_good = 0
    max_bad = -10
    for i in arena.arenas[0].items:
        if i.name in ['GoodGoal','GoodGoalBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            max_good = max(i.sizes[0].x,max_good)
        if i.name in ['BadGoal','BadGoalBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            max_bad = max(i.sizes[0].x,max_bad)        
        if i.name in ['GoodGoalMulti','GoodGoalMultiBounce']:
            if len(i.sizes)==0: #arena max cannot be computed
                return -1
            tot_reward += i.sizes[0].x  

    tot_reward += max_good
    if tot_reward == 0:
        tot_reward = max_bad  #optimal is to die
    return tot_reward


def main():
    # Load the agent from the submission
    print('Loading your agent')
    try:
        spec = importlib.util.spec_from_file_location('agent_module', '/aaio/agent.py')
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        submitted_agent = agent_module.Agent(device='cuda')
    except Exception as e:
        print('Your agent could not be loaded, make sure all the paths are absolute, error thrown:')
        raise e
    print('Agent successfully loaded')

    arenas = glob.glob('/aaio/test/train_arenas/*.yaml')
    
    #arena_config_in = ArenaConfig('/aaio/test/1-Food.yaml')

    #print('Resetting your agent')
    #try:
    #    submitted_agent.reset(t=arena_config_in.arenas[0].t)
    #except Exception as e:
    #    print('Your agent could not be reset:')
    #    raise e

    try:
        resolution = submitted_agent.resolution
        assert resolution == 84
    except AttributeError:
        resolution = 84
    except AssertionError:
        print('Resolution must be 84 for testing')
        return

    env = AnimalAIEnv(
        environment_filename='/aaio/test/env/AnimalAI',
        seed=0,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=True,
        resolution=resolution
    )

    print('Running arenas')
    total_reward = 0
    total_perf = 0
    for count,a in enumerate(arenas):
        arena_config_in = ArenaConfig(a)
        max_reward = analyze_arena(arena_config_in)
        if max_reward<0:
            print("Warning: Arena {} max_reward cannot be computed".format(a))
        env.reset(arenas_configurations=arena_config_in)
        cumulated_reward = 0

        try:
            submitted_agent.reset(t=arena_config_in.arenas[0].t)
        except Exception as e:
            print('Agent reset failed during episode {}'.format(k))
            raise e
        try:
            obs, reward, done, info = env.step([0, 0])
            for i in range(arena_config_in.arenas[0].t):

                action = submitted_agent.step(obs, reward, done, info)
                obs, reward, done, info = env.step(action)
                cumulated_reward += reward
                if done:
                    break
        except Exception as e:
            print('Episode {} failed'.format(a))
            raise e
        total_reward += cumulated_reward
        performance = cumulated_reward/max_reward
        total_perf += performance
        print('Episode {}:{} completed, reward {}:{}  performance {}:{}'.format(count,a, cumulated_reward, 
            total_reward/(count+1), performance, total_perf/(count+1)),flush=True)

    print('SUCCESS {} {}'.format(total_reward/count,total_perf/(count+1)))


if __name__ == '__main__':
    main()
