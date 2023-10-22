import os

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)