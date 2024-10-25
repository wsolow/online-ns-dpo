import os
import time

def create_log_dirs(args, config_name, config):

    log_dirs = dict()
    exp_id = time.strftime("%Y-%m-%d-%H-%M-%S")

    project_name = args.project

    for seed in range(config.seed_start, config.seed_start+config.num_seeds):
        if project_name is not None:
            log_dir = os.path.join(config.logdir, project_name, config_name, exp_id, f"{config.algo}_{seed}")
        else:
            log_dir = os.path.join(config.logdir, config_name, exp_id, f"{config.algo}_{seed}")
        os.makedirs(log_dir, exist_ok=True)
        log_dirs[seed] = log_dir

    return log_dirs

def timeit(func):
    def wrap(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print("%s use %.3f sec" % (func.__name__, te - ts))
        return result

    return wrap

# def save_code(save_dir):
#     project_dir = PROJECT_DIR
#     shutil.copytree(
#         project_dir,
#         save_dir + "/code",
#         ignore=shutil.ignore_patterns(
#             "venv",
#             "log*",
#             "result*",
#             "refs",
#             "dataset*",
#             "model*",
#             ".git",
#             "*.pyc",
#             ".idea",
#             ".DS_Store",
#         ),
#     )

