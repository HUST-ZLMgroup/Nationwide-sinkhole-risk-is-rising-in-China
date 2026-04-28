from __future__ import annotations

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


# ============================================================
# 批量运行配置
# ============================================================
CSV_NAMES: List[str] = [
    # "Positive_Negative_balanced_25366.csv",
    "Points_China_all_10km.csv",
]

SSP_LIST: List[str] = ["ssp1", "ssp3", "ssp5"]

# True: 某一组失败后立刻停止
# False: 继续执行剩余组合，并在最后汇总失败项
STOP_ON_ERROR = False


# ============================================================
# 工具函数
# ============================================================
def build_combinations() -> List[Tuple[str, str]]:
    """生成 2 × 3 = 6 组 (csv_name, ssp) 组合。"""
    return [(csv_name, ssp) for csv_name in CSV_NAMES for ssp in SSP_LIST]



def replace_df_path_block(script_text: str, csv_name: str) -> str:
    """
    替换主脚本中 df_path = os.path.join(...) 这一整段。
    仅替换首个匹配，避免影响其他 os.path.join 调用。
    """
    pattern = re.compile(
        r"df_path\s*=\s*os\.path\.join\(\n[\s\S]*?\n\)",
        flags=re.MULTILINE,
    )

    replacement = (
        'df_path = os.path.join(\n'
        '    input_folder_path, "points",\n'
        f'    "{csv_name}"\n'
        ')'
    )

    new_text, n = pattern.subn(replacement, script_text, count=1)
    if n != 1:
        raise RuntimeError("未能唯一匹配 df_path = os.path.join(...) 代码块，请检查主脚本格式。")
    return new_text



def replace_ssp_line(script_text: str, ssp: str) -> str:
    """替换主脚本中的 ssp = "..." 这一行。"""
    pattern = re.compile(r'^ssp\s*=\s*["\'][^"\']+["\'].*$', flags=re.MULTILINE)
    replacement = f'ssp = "{ssp}"  # 由批处理脚本自动写入'

    new_text, n = pattern.subn(replacement, script_text, count=1)
    if n != 1:
        raise RuntimeError("未能唯一匹配 ssp = ... 这一行，请检查主脚本格式。")
    return new_text



def build_temp_script(main_script_path: Path, csv_name: str, ssp: str, temp_dir: Path) -> Path:
    """基于原始 Main_function_Feature_Extract.py 生成单次运行的临时脚本。"""
    script_text = main_script_path.read_text(encoding="utf-8")
    script_text = replace_df_path_block(script_text, csv_name)
    script_text = replace_ssp_line(script_text, ssp)

    safe_csv_stem = Path(csv_name).stem
    temp_script_path = temp_dir / f"__tmp__Main_function_Feature_Extract__{safe_csv_stem}__{ssp}.py"
    temp_script_path.write_text(script_text, encoding="utf-8")
    return temp_script_path



def cleanup_temp_scripts(script_dir: Path) -> None:
    """删除当前目录下由本批处理脚本生成的所有临时脚本。"""
    temp_files = sorted(script_dir.glob("__tmp__Main_function_Feature_Extract__*.py"))
    if not temp_files:
        return

    for temp_file in temp_files:
        try:
            temp_file.unlink()
            print(f"[清理] 已删除临时脚本: {temp_file.name}")
        except Exception as exc:
            print(f"[警告] 删除临时脚本失败: {temp_file} | {exc}")



def run_one_case(
    python_executable: str,
    main_script_dir: Path,
    temp_script_path: Path,
    csv_name: str,
    ssp: str,
    log_dir: Path,
) -> int:
    """执行一组组合，并将 stdout/stderr 写入日志文件。"""
    log_name = f"{Path(csv_name).stem}__{ssp}.log"
    log_path = log_dir / log_name

    print("\n" + "=" * 100)
    print(f"开始执行: csv={csv_name} | ssp={ssp}")
    print(f"临时脚本: {temp_script_path.name}")
    print(f"日志文件: {log_path}")
    print("=" * 100)

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(main_script_dir)
        if not old_pythonpath
        else str(main_script_dir) + os.pathsep + old_pythonpath
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[Batch] csv_name   = {csv_name}\n")
        log_file.write(f"[Batch] ssp        = {ssp}\n")
        log_file.write(f"[Batch] script     = {temp_script_path.name}\n")
        log_file.write(f"[Batch] PYTHONPATH = {env['PYTHONPATH']}\n\n")
        log_file.flush()

        completed = subprocess.run(
            [python_executable, str(temp_script_path)],
            cwd=str(main_script_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode == 0:
        print(f"执行成功: csv={csv_name} | ssp={ssp}")
    else:
        print(f"执行失败: csv={csv_name} | ssp={ssp} | returncode={completed.returncode}")
        print(f"请查看日志: {log_path}")

    return completed.returncode


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    main_script_path = script_dir / "Main_function_Feature_Extract.py"

    if not main_script_path.exists():
        raise FileNotFoundError(
            f"未找到主脚本: {main_script_path}\n"
            f"请将本批处理脚本与 Main_function_Feature_Extract.py 放在同一文件夹下。"
        )

    # 临时脚本放在主脚本同级目录下，避免相对导入找不到同目录模块
    temp_dir = script_dir
    log_dir = script_dir / "_batch_logs"
    log_dir.mkdir(exist_ok=True)

    # 先清理历史残留的临时脚本
    cleanup_temp_scripts(script_dir)

    combinations = build_combinations()
    print(f"将要执行以下 {len(combinations)} 组组合:")
    for i, (csv_name, ssp) in enumerate(combinations, start=1):
        print(f"  {i}. {csv_name} | {ssp}")

    failed_cases: List[Tuple[str, str, int]] = []

    try:
        for csv_name, ssp in combinations:
            temp_script_path = build_temp_script(main_script_path, csv_name, ssp, temp_dir)

            try:
                returncode = run_one_case(
                    python_executable=sys.executable,
                    main_script_dir=script_dir,
                    temp_script_path=temp_script_path,
                    csv_name=csv_name,
                    ssp=ssp,
                    log_dir=log_dir,
                )
            finally:
                # 每组跑完后立即删除对应临时脚本
                try:
                    if temp_script_path.exists():
                        temp_script_path.unlink()
                        print(f"[清理] 已删除临时脚本: {temp_script_path.name}")
                except Exception as exc:
                    print(f"[警告] 删除临时脚本失败: {temp_script_path} | {exc}")

            if returncode != 0:
                failed_cases.append((csv_name, ssp, returncode))
                if STOP_ON_ERROR:
                    break
    finally:
        # 兜底清理，防止异常中断后仍有残留
        cleanup_temp_scripts(script_dir)

    print("\n" + "#" * 100)
    print("批量运行结束")
    print(f"日志目录: {log_dir}")

    if failed_cases:
        print("以下组合执行失败:")
        for csv_name, ssp, returncode in failed_cases:
            print(f"  - {csv_name} | {ssp} | returncode={returncode}")
        raise SystemExit(1)
    else:
        print(f"全部 {len(combinations)} 组组合执行成功。")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
