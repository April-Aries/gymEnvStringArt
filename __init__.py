from gym.envs.registration import register

register(
    id='StringArt-v0',
    entry_point='StringArt.envs:StringArtEnv',
    max_episode_steps=4000
)