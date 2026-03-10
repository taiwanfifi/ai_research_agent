"""
GPU Resource Manager — Vast.ai Integration
=============================================
Manages remote GPU instances for heavy compute tasks:
- Search for cheapest GPU matching requirements
- Create/start/stop/destroy instances
- Transfer code and data
- Track usage and costs for accounting

Designed for extensibility — vast.ai today, other platforms tomorrow.
All cost/usage data logged to gpu_usage.json for billing.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ── Usage tracking ─────────────────────────────────────────

USAGE_LOG_PATH = Path(__file__).parent.parent / "missions" / "_gpu_usage.json"


@dataclass
class GPUSession:
    """A single GPU rental session."""
    session_id: str
    mission_id: str
    platform: str                  # "vastai", "lambda", "runpod", etc.
    instance_id: str
    gpu_model: str                 # "RTX_4090", "A100_SXM4", etc.
    gpu_ram_gb: float
    cost_per_hour: float           # $/hr
    started_at: str                # ISO timestamp
    stopped_at: str = ""
    total_hours: float = 0.0
    total_cost: float = 0.0
    tasks_run: list = field(default_factory=list)  # [{task, duration_s, success}]
    data_transferred_mb: float = 0.0
    status: str = "active"        # active, stopped, destroyed

    def to_dict(self) -> dict:
        return asdict(self)


def _load_usage_log() -> list[dict]:
    """Load GPU usage history."""
    if USAGE_LOG_PATH.exists():
        try:
            with open(USAGE_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_usage_log(sessions: list[dict]):
    """Save GPU usage history."""
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG_PATH, "w") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def log_session(session: GPUSession):
    """Append a session to the usage log."""
    sessions = _load_usage_log()
    # Update existing or append new
    for i, s in enumerate(sessions):
        if s.get("session_id") == session.session_id:
            sessions[i] = session.to_dict()
            _save_usage_log(sessions)
            return
    sessions.append(session.to_dict())
    _save_usage_log(sessions)


def get_usage_summary() -> dict:
    """Get cumulative usage summary for billing."""
    sessions = _load_usage_log()
    total_cost = sum(s.get("total_cost", 0) for s in sessions)
    total_hours = sum(s.get("total_hours", 0) for s in sessions)
    by_gpu = {}
    for s in sessions:
        gpu = s.get("gpu_model", "unknown")
        by_gpu[gpu] = by_gpu.get(gpu, 0) + s.get("total_cost", 0)

    return {
        "total_sessions": len(sessions),
        "total_hours": round(total_hours, 2),
        "total_cost_usd": round(total_cost, 4),
        "cost_by_gpu": {k: round(v, 4) for k, v in by_gpu.items()},
        "last_session": sessions[-1] if sessions else None,
    }


# ── Vast.ai operations ────────────────────────────────────

def _run_vastai(*args, timeout: int = 30) -> str:
    """Run a vastai CLI command and return stdout."""
    cmd = ["vastai"] + list(args)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vastai error: {result.stderr.strip()}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"vastai command timed out: {' '.join(cmd)}")


def search_gpus(min_ram_gb: int = 24, max_cost_per_hour: float = 0.50,
                num_gpus: int = 1, min_reliability: float = 0.95) -> list[dict]:
    """Search for available GPU offers on vast.ai.

    Returns sorted by cost (cheapest first).
    """
    query = (
        f"gpu_ram>={min_ram_gb} "
        f"reliability>{min_reliability} "
        f"num_gpus={num_gpus} "
        f"dph<={max_cost_per_hour}"
    )
    output = _run_vastai("search", "offers", query, "-o", "dph+", "--raw", timeout=60)
    if not output:
        return []

    try:
        offers = json.loads(output)
        return [{
            "id": o.get("id"),
            "gpu_model": o.get("gpu_name", "unknown"),
            "gpu_ram_gb": o.get("gpu_ram", 0),
            "num_gpus": o.get("num_gpus", 1),
            "cost_per_hour": o.get("dph_total", 0),
            "reliability": o.get("reliability2", 0),
            "vcpus": o.get("cpu_cores_effective", 0),
            "ram_gb": o.get("cpu_ram", 0),
            "disk_gb": o.get("disk_space", 0),
            "inet_up_mbps": o.get("inet_up", 0),
            "inet_down_mbps": o.get("inet_down", 0),
        } for o in offers[:10]]
    except json.JSONDecodeError:
        # Fallback: parse tabular output
        return []


def create_instance(offer_id: int, disk_gb: int = 60,
                    image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
                    mission_id: str = "") -> dict:
    """Create a new GPU instance from an offer.

    Returns: {"instance_id": str, "session": GPUSession}
    """
    output = _run_vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--ssh", "--direct",
        timeout=60,
    )

    # Parse instance ID from output
    instance_id = ""
    for line in output.split("\n"):
        if "new contract" in line.lower() or "instance" in line.lower():
            # Try to extract ID
            import re
            match = re.search(r'\b(\d{6,})\b', line)
            if match:
                instance_id = match.group(1)
                break

    if not instance_id:
        # Try to get from show instances
        time.sleep(3)
        instances = show_instances()
        if instances:
            instance_id = str(instances[0].get("id", ""))

    session = GPUSession(
        session_id=f"gpu_{int(time.time())}",
        mission_id=mission_id,
        platform="vastai",
        instance_id=instance_id,
        gpu_model="",  # Will be filled from offer
        gpu_ram_gb=0,
        cost_per_hour=0,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    return {"instance_id": instance_id, "session": session, "raw": output}


def show_instances() -> list[dict]:
    """List current instances."""
    output = _run_vastai("show", "instances", "--raw")
    if not output:
        return []
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return []


def _parse_ssh_info(instance_id: str) -> dict:
    """Parse SSH connection info from vastai ssh-url.

    Returns: {"host": str, "port": int, "user": str}
    """
    import re
    output = _run_vastai("ssh-url", str(instance_id))
    # Format: ssh://root@<host>:<port>  or  ssh -p <port> root@<host>
    # or just: root@<host> -p <port>

    # Try ssh:// URL format
    url_match = re.search(r'ssh://(\w+)@([\w\.\-]+):(\d+)', output)
    if url_match:
        return {"user": url_match.group(1), "host": url_match.group(2), "port": int(url_match.group(3))}

    # Try -p format
    p_match = re.search(r'-p\s+(\d+)\s+(\w+)@([\w\.\-]+)', output)
    if p_match:
        return {"user": p_match.group(2), "host": p_match.group(3), "port": int(p_match.group(1))}

    # Try user@host -p port
    alt_match = re.search(r'(\w+)@([\w\.\-]+).*-p\s*(\d+)', output)
    if alt_match:
        return {"user": alt_match.group(1), "host": alt_match.group(2), "port": int(alt_match.group(3))}

    raise RuntimeError(f"Cannot parse SSH info from: {output}")


SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR"


def get_ssh_command(instance_id: str) -> str:
    """Get SSH connection command for an instance."""
    info = _parse_ssh_info(instance_id)
    return f"ssh {SSH_OPTS} -p {info['port']} {info['user']}@{info['host']}"


def wait_for_ssh(instance_id: str, max_wait: int = 300, poll_interval: int = 10) -> bool:
    """Wait until instance is SSH-ready. Adaptive polling.

    Returns True if ready, raises RuntimeError on timeout.
    """
    info = _parse_ssh_info(instance_id)
    ssh_cmd = f"ssh {SSH_OPTS} -p {info['port']} {info['user']}@{info['host']} echo ready"

    start = time.time()
    attempts = 0
    while time.time() - start < max_wait:
        attempts += 1
        try:
            result = subprocess.run(
                ssh_cmd, shell=True, capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and "ready" in result.stdout:
                elapsed = time.time() - start
                print(f"  [GPU] SSH ready after {elapsed:.0f}s ({attempts} attempts)")
                return True
        except subprocess.TimeoutExpired:
            pass

        # Adaptive backoff: start fast, slow down
        wait = min(poll_interval + attempts * 2, 30)
        print(f"  [GPU] Waiting for SSH... (attempt {attempts}, next check in {wait}s)")
        time.sleep(wait)

    raise RuntimeError(f"Instance {instance_id} not SSH-ready after {max_wait}s")


def execute_remote(instance_id: str, command: str, timeout: int = 600) -> str:
    """Execute a command on a remote instance via SSH.

    Uses SSH directly instead of vastai execute (which returns 400 on complex commands).
    """
    info = _parse_ssh_info(instance_id)
    ssh_cmd = f"ssh {SSH_OPTS} -p {info['port']} {info['user']}@{info['host']} {command}"

    try:
        result = subprocess.run(
            ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[STDERR]: {result.stderr}" if result.stderr else ""
        return output.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SSH command timed out after {timeout}s")


def stop_instance(instance_id: str) -> str:
    """Stop (pause) an instance. Keeps data, storage fee only."""
    return _run_vastai("stop", "instance", str(instance_id))


def destroy_instance(instance_id: str) -> str:
    """Destroy an instance. IRREVERSIBLE — deletes all data."""
    return _run_vastai("destroy", "instance", str(instance_id))


def transfer_files(local_path: str, instance_id: str, remote_path: str,
                   direction: str = "upload") -> str:
    """Transfer files to/from an instance via SCP.

    direction: "upload" (local→remote) or "download" (remote→local)
    """
    info = _parse_ssh_info(instance_id)
    scp_opts = f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P {info['port']}"

    if direction == "upload":
        cmd = f"scp {scp_opts} -r {local_path} {info['user']}@{info['host']}:{remote_path}"
    else:
        cmd = f"scp {scp_opts} -r {info['user']}@{info['host']}:{remote_path} {local_path}"

    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        return f"[SCP ERROR]: {result.stderr.strip()}"
    return result.stdout + result.stderr


# ── High-level workflow ────────────────────────────────────

def quick_gpu_run(code: str, mission_id: str = "",
                  min_ram_gb: int = 24, max_cost: float = 0.35,
                  timeout_minutes: int = 30) -> dict:
    """Complete workflow: find GPU → create → run code → get results → destroy.

    For short tasks (training, inference). Handles full lifecycle.

    Args:
        code: Python code to execute on GPU
        mission_id: for cost tracking
        min_ram_gb: minimum GPU RAM
        max_cost: maximum $/hr
        timeout_minutes: max runtime before auto-destroy

    Returns:
        {"success": bool, "output": str, "cost": float, "gpu": str}
    """
    print(f"  [GPU] Searching for GPU (≥{min_ram_gb}GB, ≤${max_cost}/hr)...")

    # 1. Find cheapest GPU
    offers = search_gpus(min_ram_gb=min_ram_gb, max_cost_per_hour=max_cost)
    if not offers:
        return {"success": False, "output": "No GPU offers found matching criteria",
                "cost": 0, "gpu": "none"}

    offer = offers[0]
    print(f"  [GPU] Found: {offer['gpu_model']} ({offer['gpu_ram_gb']}GB) "
          f"@ ${offer['cost_per_hour']:.4f}/hr")

    # 2. Create instance
    print(f"  [GPU] Creating instance...")
    result = create_instance(offer["id"], mission_id=mission_id)
    instance_id = result["instance_id"]
    session = result["session"]
    session.gpu_model = offer["gpu_model"]
    session.gpu_ram_gb = offer["gpu_ram_gb"]
    session.cost_per_hour = offer["cost_per_hour"]

    try:
        # 3. Wait for instance to be running (vast.ai status)
        print(f"  [GPU] Waiting for instance {instance_id} to start...")
        for attempt in range(60):  # Max 5 min wait for status
            time.sleep(5)
            instances = show_instances()
            for inst in instances:
                if str(inst.get("id")) == str(instance_id):
                    status = inst.get("actual_status", "")
                    if status == "running":
                        print(f"  [GPU] Instance status: running")
                        break
                    elif status in ("exited", "error"):
                        raise RuntimeError(f"Instance failed with status: {status}")
            else:
                continue
            break
        else:
            raise RuntimeError("Instance failed to start within 5 minutes")

        # 4. Wait for SSH readiness (adaptive — instance may need time after "running")
        wait_for_ssh(instance_id, max_wait=180)

        # 5. Upload code via SCP
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, prefix='gpu_task_',
        ) as f:
            f.write(code)
            local_script = f.name

        print(f"  [GPU] Uploading code ({len(code)} chars)...")

        try:
            upload_result = transfer_files(local_script, instance_id, "/tmp/gpu_task.py", "upload")
            if "[SCP ERROR]" in upload_result:
                # Fallback: pipe code via SSH stdin
                import base64
                code_b64 = base64.b64encode(code.encode()).decode()
                if len(code_b64) < 50000:
                    execute_remote(
                        instance_id,
                        f"'echo {code_b64} | base64 -d > /tmp/gpu_task.py'",
                        timeout=30,
                    )
                else:
                    raise RuntimeError(f"SCP failed and code too large for base64: {upload_result}")
        finally:
            os.unlink(local_script)

        # 6. Install deps + run via SSH
        print(f"  [GPU] Installing dependencies & executing...")

        # Use bash -c to chain commands properly over SSH
        run_cmd = (
            "'pip install transformers datasets peft accelerate scipy matplotlib -q 2>/dev/null "
            "&& python3 /tmp/gpu_task.py'"
        )
        output = execute_remote(
            instance_id,
            run_cmd,
            timeout=timeout_minutes * 60,
        )

        # 5. Calculate cost
        elapsed_hours = (time.time() - time.mktime(
            time.strptime(session.started_at, "%Y-%m-%dT%H:%M:%S")
        )) / 3600
        session.total_hours = round(elapsed_hours, 4)
        session.total_cost = round(elapsed_hours * session.cost_per_hour, 4)
        session.status = "completed"
        session.stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        session.tasks_run.append({
            "task": "quick_gpu_run",
            "duration_s": round(elapsed_hours * 3600),
            "success": True,
        })

        return {
            "success": True,
            "output": output,
            "cost": session.total_cost,
            "gpu": offer["gpu_model"],
            "hours": session.total_hours,
        }

    except Exception as e:
        session.status = "error"
        session.stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        elapsed_hours = (time.time() - time.mktime(
            time.strptime(session.started_at, "%Y-%m-%dT%H:%M:%S")
        )) / 3600
        session.total_hours = round(elapsed_hours, 4)
        session.total_cost = round(elapsed_hours * session.cost_per_hour, 4)
        return {
            "success": False,
            "output": str(e),
            "cost": session.total_cost,
            "gpu": offer.get("gpu_model", "unknown"),
        }

    finally:
        # ALWAYS destroy to avoid runaway costs
        print(f"  [GPU] Destroying instance {instance_id}...")
        try:
            destroy_instance(instance_id)
        except Exception:
            print(f"  [GPU] WARNING: Failed to destroy instance {instance_id}! "
                  f"Check vast.ai dashboard!")

        # Log session
        log_session(session)
        print(f"  [GPU] Session logged: ${session.total_cost:.4f} "
              f"({session.total_hours:.2f}h on {session.gpu_model})")
