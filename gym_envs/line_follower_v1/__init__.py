from gymnasium.envs.registration import register

register(
    id="my_gym_envs/line_follower_v1",
    entry_point="line_follower_v1.envs:LineFollowerEnv",
)
