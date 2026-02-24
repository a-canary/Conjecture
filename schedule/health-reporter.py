# cron: */4 * * * *
# timeout: 30
# approval: auto
"""Health reporter — writes system stats to /data/dashboard/health.json.

Deterministic script. No Claude, no network. Reads /proc for CPU/memory,
shutil for disk, ps for processes.
"""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HEALTH_FILE = Path("/data/dashboard/health.json")
INTERESTING_PROCS = {"claude", "node", "python3", "git", "npm", "npx", "pytest"}


def get_cpu_percent() -> float:
    """Read CPU usage from /proc/stat (container-scoped)."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        fields = line.split()[1:]
        total = sum(int(f) for f in fields)
        idle = int(fields[3])
        prev_file = Path("/tmp/.health-cpu-prev")
        if prev_file.exists():
            prev = json.loads(prev_file.read_text())
            d_total = total - prev["total"]
            d_idle = idle - prev["idle"]
            cpu_pct = ((d_total - d_idle) / d_total * 100) if d_total > 0 else 0.0
        else:
            cpu_pct = 0.0
        prev_file.write_text(json.dumps({"total": total, "idle": idle}))
        return round(cpu_pct, 1)
    except (OSError, IndexError, ZeroDivisionError):
        return 0.0


def get_memory() -> dict:
    """Read memory from /proc/meminfo."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if parts[0] in ("MemTotal:", "MemAvailable:"):
                    info[parts[0].rstrip(":")] = int(parts[1])
        total_mb = info.get("MemTotal", 0) / 1024
        avail_mb = info.get("MemAvailable", 0) / 1024
        used_mb = total_mb - avail_mb
        pct = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
        return {"used_mb": round(used_mb), "total_mb": round(total_mb), "percent": round(pct, 1)}
    except (OSError, KeyError, ZeroDivisionError):
        return {"used_mb": 0, "total_mb": 0, "percent": 0.0}


def get_disk() -> dict:
    """Disk usage for key paths."""
    result = {}
    for path in ("/workspace", "/data"):
        try:
            usage = shutil.disk_usage(path)
            result[path] = {
                "used_gb": round(usage.used / (1024 ** 3), 2),
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "percent": round(usage.used / usage.total * 100, 1),
            }
        except OSError:
            result[path] = {"used_gb": 0, "total_gb": 0, "percent": 0.0}
    return result


def get_processes() -> list[dict]:
    """List interesting processes with CPU/MEM stats."""
    try:
        out = subprocess.run(
            ["ps", "aux", "--no-headers"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return []

    mem_info = get_memory()
    total_mb = mem_info["total_mb"]
    procs = []
    for line in out.strip().splitlines():
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        cmd = parts[10]
        cmd_base = cmd.split()[0].split("/")[-1] if cmd else ""
        if cmd_base in INTERESTING_PROCS:
            procs.append({
                "name": cmd_base,
                "pid": int(parts[1]),
                "cpu_percent": float(parts[2]),
                "mem_mb": round(float(parts[3]) * total_mb / 100),
                "command": cmd[:120],
            })
    return procs


if __name__ == "__main__":
    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": get_cpu_percent(),
        "memory": get_memory(),
        "disk": get_disk(),
        "processes": get_processes(),
    }
    HEALTH_FILE.write_text(json.dumps(report, indent=2))
