import torch
import subprocess
import datetime

def _smi(attr: str) -> list:
    output = subprocess.run(
        ["nvidia-smi", f"--query-gpu={attr}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True
    )
    return output.stdout.strip().split("\n")

def vram_snapshot(tag: str, log_file: str = "vram.log") -> dict:

    kernel = round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0
    try:
        used = round(float(_smi("memory.used")[0]) / 1024.0, 2)
    except Exception:
        used = 0.0

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [vram] {tag:22} gpu_used={used}GB  kernel={kernel}GB"

    print(log_msg)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    return {"timestamp": timestamp, "tag": tag, "gpu_used_gb": used, "kernel_gb": kernel}


