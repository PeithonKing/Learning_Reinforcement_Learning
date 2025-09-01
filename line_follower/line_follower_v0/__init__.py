from gymnasium.envs.registration import register

register(
    id="gymnasium_env/line_follower_v0",
    entry_point="line_follower_v0.envs:LineFollowerEnv",
)
