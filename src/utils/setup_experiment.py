import os, datetime


def setup_experiment(model_name):
    exp_dir = os.path.join("experiments", model_name)
    os.makedirs(exp_dir, exist_ok=True)
    subdirs = ['results', 'boxes', 'heatmaps']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    return exp_dir

def get_exp_dir(model_name):
    return os.path.abspath(os.path.join("experiments", model_name))


def get_best_weights(model_name):
    """Locates the best.pt file for a given model name."""
    weights_path = os.path.join("experiments", model_name, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ No weights found at {weights_path}. Did you finish training?")
    return weights_path
