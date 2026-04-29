# run_batch_notebooks.py
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# ====== You need to change the path according to the actual situation ======
# NOTEBOOK_PATH = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/compute_climate_change_increment.ipynb") # Run the result code
NOTEBOOK_PATH = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/plot_climate_change_increment.ipynb")   # Processing step.
OUTPUT_DIR = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/batch_outputs")                          # Output directory (will be created automatically)
WORK_DIR = NOTEBOOK_PATH.parent                             # Working directory when notebook is executed
KERNEL_NAME = "python3"                                     # Generally defaults to python3

ssps = ["ssp2"]
ssp_times = ["2040", "2060", "2080", "2100"]
# ssp_times = ["2080", "2100"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_params_cell(nb, ssp_value: str, ssp_time_value: str) -> None:
    """Set/replace parameter cells in notebook:
    - If a code unit containing both 'ssp =' and 'ssp_time =' is found, replace its content
    - Otherwise insert a new code unit at the front"""
    param_src = f'ssp = "{ssp_value}"\nssp_time = "{ssp_time_value}"\n'
    target_idx = None

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source or ""
        if ("ssp" in src) and ("ssp_time" in src) and ("=" in src):
            # More strictly: include both ssp = and ssp_time =
            if ("ssp" in src and "ssp_time" in src and "ssp =" in src and "ssp_time" in src):
                target_idx = i
                break

    if target_idx is not None:
        nb.cells[target_idx].source = param_src
    else:
        nb.cells.insert(0, nbformat.v4.new_code_cell(param_src))

def run_one(ssp_value: str, ssp_time_value: str):
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    set_params_cell(nb, ssp_value, ssp_time_value)

    ep = ExecutePreprocessor(
        timeout=-1,              # ()
        kernel_name=KERNEL_NAME,
        allow_errors=False       # If an error is reported, interrupt the combination (more conducive to locating the problem)
    )

    print(f"[RUN] ssp={ssp_value}, ssp_time={ssp_time_value}")
    ep.preprocess(nb, {"metadata": {"path": str(WORK_DIR)}})

    out_path = OUTPUT_DIR / f"{NOTEBOOK_PATH.stem}_{ssp_value}_{ssp_time_value}.ipynb"
    nbformat.write(nb, out_path)
    print(f"[OK ] saved -> {out_path}")

def main():
    for ssp_value in ssps:
        for ssp_time_value in ssp_times:
            run_one(ssp_value, ssp_time_value)

if __name__ == "__main__":
    main()
