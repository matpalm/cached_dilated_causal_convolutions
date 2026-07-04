from pathlib import Path


def model_data_z_path_for(run: str, check_exists: bool = True):
    p = (
        Path(__file__).parent.parent
        / "parametric_capture"
        / "runs"
        / run
        / "model_data.z"
    )
    if check_exists and (not p.exists()):
        raise Exception(f"model_data.z path [{p}] doesn't exist?")
    return str(p)
