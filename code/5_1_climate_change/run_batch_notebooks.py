# run_batch_notebooks.py
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# ====== 需要你按实际情况改一下路径 ======
# NOTEBOOK_PATH = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/compute_climate_change_increment.ipynb")   # 跑结果的代码
NOTEBOOK_PATH = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/plot_climate_change_increment.ipynb")   # 出图的代码
OUTPUT_DIR = Path("/path/to/sinkhole-risk-china/code/5_1_climate_change/batch_outputs")                          # 输出目录（会自动创建）
WORK_DIR = NOTEBOOK_PATH.parent                             # notebook 执行时的工作目录
KERNEL_NAME = "python3"                                     # 一般默认 python3

ssps = ["ssp2"]
ssp_times = ["2040", "2060", "2080", "2100"]
# ssp_times = ["2080", "2100"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def set_params_cell(nb, ssp_value: str, ssp_time_value: str) -> None:
    """
    在 notebook 中设置/替换参数单元：
    - 若找到同时包含 'ssp =' 与 'ssp_time =' 的代码单元，则替换其内容
    - 否则在最前面插入一个新的代码单元
    """
    param_src = f'ssp = "{ssp_value}"\nssp_time = "{ssp_time_value}"\n'
    target_idx = None

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = cell.source or ""
        if ("ssp" in src) and ("ssp_time" in src) and ("=" in src):
            # 更严格一点：同时包含 ssp = 与 ssp_time =
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
        timeout=-1,              # 不设超时（按需也可改成秒数）
        kernel_name=KERNEL_NAME,
        allow_errors=False       # 遇到报错就中断该组合（更利于定位问题）
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
