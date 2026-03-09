import os


def setup_experiment(model_name):
    exp_dir = os.path.join("experiments", f"{model_name}")
    os.makedirs(exp_dir, exist_ok=True)
    subdirs = ['results', 'boxes', 'heatmaps']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    return exp_dir

def get_exp_dir(model_name):
    return os.path.abspath(os.path.join("experiments", f"{model_name}"))
