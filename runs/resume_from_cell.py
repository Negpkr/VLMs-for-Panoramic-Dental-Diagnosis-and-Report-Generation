#!/usr/bin/env python3
"""Resume llava_evaluation.ipynb from a given cell index (default 18)."""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTHONPATH"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path("/home/s222393187/Dental/llava_evaluation.ipynb")
RUN_DIR = Path("/home/s222393187/Dental/runs")
RUN_DIR.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
START_CELL = int(os.environ.get("START_CELL", "18"))
LOG = RUN_DIR / f"resume_from_{START_CELL}_{STAMP}.log"
OUT_NB = RUN_DIR / f"llava_evaluation_executed_from{START_CELL}_{STAMP}.ipynb"
STATUS = RUN_DIR / "CURRENT_STATUS.txt"
PID_FILE = RUN_DIR / "CURRENT_RUN.pid"

# Prior checkpoint to warm-start definitions if resuming mid-notebook
WARM_NB = Path(
    os.environ.get(
        "WARM_NOTEBOOK",
        "/home/s222393187/Dental/runs/llava_evaluation_executed_20260721_192554.ipynb",
    )
)


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")
    STATUS.write_text(line + "\n")


def preview(source: str, n: int = 90) -> str:
    s = " ".join(source.strip().split())
    return (s[:n] + "…") if len(s) > n else s


def main() -> int:
    PID_FILE.write_text(str(os.getpid()))
    log(f"RESUME from cell {START_CELL} pid={os.getpid()}")
    log(f"notebook={NOTEBOOK}")
    log(f"log={LOG}")
    log(f"output={OUT_NB}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Use latest source notebook (has the model-load fix)
    nb = nbformat.read(NOTEBOOK, as_version=4)

    # Optionally copy prior outputs for cells before START_CELL (for the saved artifact)
    if WARM_NB.exists():
        try:
            warm = nbformat.read(WARM_NB, as_version=4)
            for i in range(min(START_CELL, len(nb.cells), len(warm.cells))):
                if nb.cells[i].cell_type == "code" and warm.cells[i].cell_type == "code":
                    nb.cells[i]["outputs"] = warm.cells[i].get("outputs", [])
                    if "execution_count" in warm.cells[i]:
                        nb.cells[i]["execution_count"] = warm.cells[i].get("execution_count")
            log(f"warmed outputs from {WARM_NB.name} for cells < {START_CELL}")
        except Exception as e:
            log(f"warm notebook skip: {e}")

    # Cells to execute: all code cells from START_CELL onward,
    # PLUS earlier setup code cells needed for definitions (0..START_CELL-1 code)
    setup_code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and i < START_CELL]
    run_code = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and i >= START_CELL]
    # Re-run setup then resume target cells (setup is fast; model load is at START_CELL)
    exec_order = setup_code + run_code
    log(f"will execute {len(exec_order)} code cells (setup {len(setup_code)} + from-{START_CELL} {len(run_code)})")

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name="dental-llava",
        allow_errors=True,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )

    t0 = time.time()
    try:
        with client.setup_kernel():
            log("kernel started")
            for n, idx in enumerate(exec_order, 1):
                cell = nb.cells[idx]
                src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
                phase = "SETUP" if idx < START_CELL else "RESUME"
                log(f"{phase} CELL {idx} ({n}/{len(exec_order)}) BEGIN | {preview(src)}")
                cell_t0 = time.time()
                try:
                    client.execute_cell(cell, idx)
                    elapsed = time.time() - cell_t0
                    err = None
                    for out in cell.get("outputs", []):
                        if out.get("output_type") == "error":
                            err = f"{out.get('ename')}: {out.get('evalue')}"
                            break
                    if err:
                        log(f"CELL {idx} DONE with ERROR after {elapsed/60:.1f} min | {err}")
                    else:
                        log(f"CELL {idx} DONE OK after {elapsed/60:.1f} min")
                except CellExecutionError as e:
                    elapsed = time.time() - cell_t0
                    log(f"CELL {idx} FAILED after {elapsed/60:.1f} min | {e}")
                except Exception as e:
                    elapsed = time.time() - cell_t0
                    log(f"CELL {idx} EXCEPTION after {elapsed/60:.1f} min | {e}")
                    log(traceback.format_exc())

                if n % 3 == 0 or n == len(exec_order):
                    nbformat.write(nb, OUT_NB)
                    log(f"checkpoint saved -> {OUT_NB.name}")

        nbformat.write(nb, OUT_NB)
        total = time.time() - t0
        log(f"FINISHED resume-from-{START_CELL} in {total/3600:.2f} hours ({total/60:.1f} min)")
        log(f"executed notebook: {OUT_NB}")
        return 0
    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        try:
            nbformat.write(nb, OUT_NB)
        except Exception:
            pass
        return 1
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
