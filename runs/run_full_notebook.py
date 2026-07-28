#!/usr/bin/env python3
"""Execute llava_evaluation.ipynb cell-by-cell with a live progress log."""
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
LOG = RUN_DIR / f"full_run_{STAMP}.log"
OUT_NB = RUN_DIR / f"llava_evaluation_executed_{STAMP}.ipynb"
STATUS = RUN_DIR / "CURRENT_STATUS.txt"
PID_FILE = RUN_DIR / "CURRENT_RUN.pid"


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
    log(f"START full notebook run pid={os.getpid()}")
    log(f"notebook={NOTEBOOK}")
    log(f"log={LOG}")
    log(f"output={OUT_NB}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    code_idxs = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    log(f"total_cells={len(nb.cells)} code_cells={len(code_idxs)}")

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
            code_done = 0
            for idx, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                code_done += 1
                src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
                log(
                    f"CELL {idx}/{len(nb.cells)-1} code {code_done}/{len(code_idxs)} BEGIN | {preview(src)}"
                )
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

                if code_done % 3 == 0 or code_done == len(code_idxs):
                    nbformat.write(nb, OUT_NB)
                    log(f"checkpoint saved -> {OUT_NB.name}")

        nbformat.write(nb, OUT_NB)
        total = time.time() - t0
        log(f"FINISHED all cells in {total/3600:.2f} hours ({total/60:.1f} min)")
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
