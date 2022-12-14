import itertools


def parse_key(key_str):
    game, frames, percentile, alar_perc, queue_size = key_str.split("-")

    return game, frames, percentile, alar_perc, queue_size

BASH_HEADER = "#!/usr/bin/env bash"
FILENAME = "detection_experiments.sh"
SEED = 1234

## All the possible scenarios
ENV_NAMES = ["Pong", "Freeway"]
ADVERSARIES = ["none", "uaps", "uapo", "osfwu"]


AGENTS = ["a2c", "dqn", "ppo"]

ATTACK_EPS = "0.01"

key_val = {
    "Freeway": {
        "dqn": '12-400-95-0.8-100',
        "a2c": '12-400-100-1.0-200',
        "ppo": '12-200-90-0.9-200'
    },
    "Pong": {
        "dqn": '12-400-100-0.9-200',
        "a2c": '12-400-100-0.9-200',
        "ppo": '12-400-100-0.9-200'

    }
}


SKIPPED_FRAMES = [str(x) for x in range(100, 701, 100)]
PERCENTILES = [str(x) for x in range(80, 101, 10)]
ALARM_PERCENTAGES = [str(x/100) for x in range(80, 101, 10)]
QUEUE_SIZES = ["100",  "200", "500"]
DETECTION_GAMES_PLAYS = ["12"]

TOTAL_GAME_PLAYS = 10

with open(FILENAME, "w") as f:
    f.write(BASH_HEADER)

    for env, adv, agent in itertools.product(ENV_NAMES, ADVERSARIES, AGENTS):
        f.write("\n#### {} {} {} #####\n".format(env, adv, agent))


        base_line = " ".join(["python main.py", "--game-mode test", "--env-name", env,
                        "--adversary", adv, "--victim-agent-mode", agent, "--detection-method KL"])

        if env == "Freeway":
            base_line = " ".join([base_line, "--allow-early-resets True"])

        base_line = " ".join([base_line, "--load-from", "trained_agents/"+ env + "/" + agent + "/model.pt"])

        for det_game in DETECTION_GAMES_PLAYS:

            if adv == "none":
                training_line = " ".join([base_line, "--detection-method-train",
                                          "--detection-game-plays", det_game, "--seed", str(SEED)])
                f.write(training_line + "\n")

            key_str = key_val[env][agent]
            game, skip_frame, percentile, alarm_perc, queue_size = parse_key(key_str)

            if adv =="none":
                game_plays = TOTAL_GAME_PLAYS
            else:
                game_plays = TOTAL_GAME_PLAYS

            test_line = " ".join([base_line, "--detection-game-plays", det_game, "--total-game-plays",
                                    str(game_plays),
                                    "--skipped-frames", skip_frame, "--percentile", percentile,
                                    "--alarm-percentage", alarm_perc, "--queue-size", queue_size,
                                    "--seed", str(SEED + 20), "--save-detection-scores"])

            f.write(test_line + "\n")
            print(key_str)
            print(test_line)

            f.write("\n")














