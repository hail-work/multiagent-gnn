import gym
from gym.envs.registration import register



from gym_multigrid.envs import soccer_game




world = soccer_game.SoccerGame4HEnv10x15N2()
register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )


register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )


world =  gym.make('multigrid-soccer-v0')
world.partial_obs = False
world.reset()


print('')
