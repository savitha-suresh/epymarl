
from gymnasium import register

def register_envs():
    sizes = (10, 15)
    p = 4
    f = 4
    o = 1
    c = False
    for s in sizes:
        register(
                id="Foraging-{0}x{0}-{1}p-{2}f-{3}s-v3".format(s, p, f, o),
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": p,
                    "min_player_level": 1,
                    "max_player_level": p,
                    "min_food_level": 1,
                    "max_food_level": p,
                    "field_size": (s, s),
                    "max_num_food": f,
                    "sight": o,
                    "max_episode_steps": 50,
                    "force_coop": c,
                },
        )