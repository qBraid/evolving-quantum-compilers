"""Run the search locally, run the model on a qBraid GPU — as one context manager.

The orchestrator (evolution loop, program database, evaluator) stays on whatever
instance you are already using, including a free CPU one. The LLM runs on an
on-demand GPU that this module brings up, serves, tunnels back to
``localhost``, and **terminates on the way out** — including on exception or
Ctrl-C.

    from qbraid_remote_gpu import RemoteGPUEndpoint

    with RemoteGPUEndpoint(profile="gpu-l40s", model="Qwen/Qwen2.5-Coder-14B-Instruct") as gpu:
        subprocess.run([
            "python", "run_evolution.py", "--endpoint", "local",
            "--base-url", gpu.base_url, "--model", gpu.model,
            "--generations", "40",
        ], check=True)
    # instance is gone here

Or from the shell::

    python qbraid_remote_gpu.py --profile gpu-l40s --generations 40

Cost discipline is structural, not advisory. Three independent stops:

1. ``max_session_minutes`` — a hard server-side ceiling, applied *before* the
   instance is even ready. Survives this process being killed outright.
2. ``auto_stop_idle_minutes`` — server-side idle stop.
3. ``__exit__`` — terminates on any exit path.

Only (1) protects you if the machine running this dies, which is exactly when a
forgotten GPU gets expensive. Terminate, not stop: a stopped instance still
bills for its disk.
"""

from __future__ import annotations

import argparse
import atexit
import pathlib
import contextlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from qbraid_core.services.compute.client import ComputeClient

# vLLM ships a compiled extension linked against a specific CUDA runtime, and a
# CUDA runtime needs a driver new enough to support it. qBraid's GPU images do
# NOT all carry the same driver: an A10 measured 570.148.08 (CUDA 12.8), while an
# L4 was older still. Current vllm wheels link libcudart.so.13 (CUDA 13), which
# needs driver >= 580, so on those images stock `pip install vllm` dies with
# either "The NVIDIA driver on your system is too old" or
# "ImportError: libcudart.so.13: cannot open shared object file".
#
# Pointing pip at a cu128 torch index does NOT fix this -- it changes which torch
# is resolved while leaving vllm's own binary built for CUDA 13, which is a worse
# failure. The version of vllm is the thing that has to match the driver:
#
#   vllm >= 0.20   torch 2.11+   CUDA 13    driver >= 580
#   vllm 0.17-0.19 torch 2.10.0  CUDA 12.8  driver >= 570
#   vllm 0.14-0.16 torch 2.9.1   CUDA 12.8  driver >= 570
#
# "auto" resolves this from the driver at launch. Override with --vllm-spec.
DEFAULT_VLLM_SPEC = "auto"

# (minimum driver major, vllm pip spec), newest first.
DRIVER_TO_VLLM = (
    (580, "vllm"),
    (570, "vllm==0.19.1"),
    (0, "vllm==0.19.1"),
)

DEFAULT_PROFILE = "gpu-l40s"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
REMOTE_PORT = 8000
LOCAL_PORT = 8000

# Substrings that mean the server is never coming up. Checked against the tail of
# vllm.log so a dead server aborts in ~30s rather than burning the full timeout.
FATAL_LOG_MARKERS = (
    "Error while finding module specification",
    "ModuleNotFoundError",
    "No module named",
    "Traceback (most recent call last)",
    "driver on your system is too old",
    "libcudart.so",
    "cannot open shared object file",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "does not appear to have a file named",
    "Repository Not Found",
    "address already in use",
)


def _log(message: str) -> None:
    print(f"[remote-gpu] {message}", flush=True)


def _instance_id(instance) -> Optional[str]:
    """Read the id off a BMAInstance without assuming the field name."""
    for attribute in ("instance_id", "instanceId", "id"):
        value = getattr(instance, attribute, None)
        if value:
            return str(value)
    return None


class RemoteGPUEndpoint:
    """A GPU instance serving an OpenAI-compatible endpoint on ``localhost``."""

    def __init__(
        self,
        profile: str = DEFAULT_PROFILE,
        model: str = DEFAULT_MODEL,
        local_port: int = LOCAL_PORT,
        max_session_minutes: int = 120,
        auto_stop_idle_minutes: int = 20,
        serve_timeout: float = 2400.0,
        keep_alive: bool = False,
        vllm_spec: str = DEFAULT_VLLM_SPEC,
    ) -> None:
        self.profile = profile
        self.model = model
        self.local_port = local_port
        self.max_session_minutes = max_session_minutes
        self.auto_stop_idle_minutes = auto_stop_idle_minutes
        self.serve_timeout = serve_timeout
        self.keep_alive = keep_alive
        self.vllm_spec = vllm_spec

        self.client = ComputeClient()
        self.instance_id: Optional[str] = None
        self.alias: Optional[str] = None
        self._tunnel: Optional[subprocess.Popen] = None
        self._provisioned = False
        self._pre_existing: set = set()
        self.server_pid: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.local_port}/v1"

    # ---- ssh helpers ----------------------------------------------------

    def _ssh(self, command: str, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             self.alias, command],
            capture_output=True, text=True, timeout=timeout,
        )

    # ---- lifecycle ------------------------------------------------------

    def __enter__(self) -> "RemoteGPUEndpoint":
        try:
            self._launch()
            self._serve()
            self._tunnel_up()
            self._await_endpoint()
        except BaseException:
            # Pull diagnostics back before the disk disappears, then make sure
            # nothing is left running.
            with contextlib.suppress(Exception):
                self.save_server_log()
            self._teardown()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._teardown()
        return False

    def _launch(self) -> None:
        # Snapshot first. If provisioning succeeds but anything afterwards
        # raises before we have an id, this is how teardown still finds the
        # machine we just started -- otherwise it bills until the session cap.
        try:
            self._pre_existing = {
                _instance_id(i) for i in self.client.list_bma_instances()
            }
        except Exception:  # noqa: BLE001 - never block a launch on this
            self._pre_existing = set()

        _log(f"launching {self.profile} ...")
        self._provisioned = True
        instance = self.client.provision_bma_instance(self.profile)
        self.instance_id = _instance_id(instance)
        if self.instance_id is None:
            raise RuntimeError(
                "provisioned an instance but could not read its id; "
                "teardown will sweep for it"
            )
        atexit.register(self._terminate_by_id, self.instance_id)
        _log(f"instance {self.instance_id}")

        # Before anything else: a server-side ceiling that outlives this process.
        self.client.update_bma_cutoff(
            self.instance_id,
            auto_stop_idle_minutes=self.auto_stop_idle_minutes,
            max_session_minutes=self.max_session_minutes,
        )
        _log(
            f"guardrails set: hard stop {self.max_session_minutes} min, "
            f"idle stop {self.auto_stop_idle_minutes} min"
        )

        self.client.wait_for_bma_instance(self.instance_id, timeout=900)
        self.client.configure_ssh_for_instance(self.instance_id)
        self.alias = self.client.bma_ssh_alias(self.instance_id)
        _log(f"ssh alias {self.alias}")

        for attempt in range(30):
            if self._ssh("echo ok", timeout=30).stdout.strip() == "ok":
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"cannot ssh to {self.alias}")

        gpu = self._ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        _log(f"gpu: {gpu.stdout.strip() or 'unknown'}")

    def _resolve_vllm_spec(self) -> str:
        """Pick a vllm release whose CUDA runtime this driver can load."""
        if self.vllm_spec != "auto":
            return self.vllm_spec

        result = self._ssh(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1",
            timeout=60,
        )
        raw = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        try:
            major = int(raw.split(".")[0])
        except (ValueError, IndexError):
            _log(f"could not parse driver version {raw!r}; assuming CUDA 12.8")
            return "vllm==0.19.1"
        for minimum, spec in DRIVER_TO_VLLM:
            if major >= minimum:
                _log(f"driver {raw} -> {spec}")
                if major < 570:
                    _log("WARNING: driver is older than any vllm we know works here")
                return spec
        return "vllm==0.19.1"

    def _ensure_vllm(self) -> str:
        """Return the path of a remote interpreter that can ``import vllm``.

        Never pipe the install through ``tail``: the pipeline's exit status is
        the *last* command's, so a failed pip looks like a success and the only
        symptom is a server that never starts.

        The system interpreter is PEP 668 externally managed on qBraid images, so
        a plain ``pip install`` there fails. Prefer an interpreter that already
        has vllm (some GPU images ship it); otherwise build a venv.
        """
        probe = self._ssh(
            "for p in python3 /usr/bin/python3; do "
            "  $p -c 'import vllm,sys; print(sys.executable)' 2>/dev/null && exit 0; "
            "done; exit 1",
            timeout=120,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            interpreter = probe.stdout.strip().splitlines()[-1]
            _log(f"vllm already present at {interpreter}")
            return interpreter

        spec = self._resolve_vllm_spec()

        venv = "$HOME/vllm-env"
        _log(f"building a venv and installing {spec} ...")
        build = self._ssh(
            f"set -e; rm -rf {venv}; python3 -m venv {venv}; "
            f"{venv}/bin/python -m pip install --upgrade pip >/dev/null 2>&1; "
            f"{venv}/bin/python -m pip install {shlex.quote(spec)}",
            timeout=3600,
        )
        if build.returncode != 0:
            tail = (build.stdout + build.stderr).strip().splitlines()[-15:]
            raise RuntimeError("vllm install failed:\n" + "\n".join(tail))

        check = self._ssh(f"{venv}/bin/python -c 'import vllm,sys; print(sys.executable)'", timeout=180)
        if check.returncode != 0:
            raise RuntimeError(f"vllm installed but will not import:\n{check.stderr.strip()}")
        interpreter = check.stdout.strip().splitlines()[-1]
        _log(f"vllm installed at {interpreter}")
        return interpreter

    def _serve(self) -> None:
        interpreter = self._ensure_vllm()

        _log(f"starting vllm for {self.model} ...")
        remote_cmd = (
            f"rm -f /tmp/vllm.log /tmp/vllm.pid; "
            f"nohup {interpreter} -m vllm.entrypoints.openai.api_server "
            f"--model {shlex.quote(self.model)} --port {REMOTE_PORT} "
            f"--tensor-parallel-size $(nvidia-smi -L | wc -l) "
            f"> /tmp/vllm.log 2>&1 & echo $! > /tmp/vllm.pid; cat /tmp/vllm.pid"
        )
        started = self._ssh(remote_cmd, timeout=180)
        self.server_pid = started.stdout.strip().splitlines()[-1] if started.stdout.strip() else None
        _log(f"server pid {self.server_pid or 'unknown'}")

    def _server_alive(self) -> bool:
        """False once the remote vllm process is gone."""
        if not self.server_pid:
            return True  # cannot tell; do not abort on that alone
        result = self._ssh(f"kill -0 {self.server_pid} 2>/dev/null && echo up || echo down", timeout=60)
        return "down" not in result.stdout

    def _tunnel_up(self) -> None:
        _log(f"tunnelling localhost:{self.local_port} -> {self.alias}:{REMOTE_PORT}")
        self._tunnel = subprocess.Popen(
            ["ssh", "-N", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
             "-L", f"{self.local_port}:localhost:{REMOTE_PORT}", self.alias],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _await_endpoint(self) -> None:
        _log("waiting for the model to load (weights download on first run) ...")
        deadline = time.time() + self.serve_timeout
        last_note = 0.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5):
                    _log("endpoint is live")
                    return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            if time.time() - last_note > 30:
                last_note = time.time()
                tail = self._ssh("tail -5 /tmp/vllm.log", timeout=30).stdout.strip()
                elapsed = int(self.serve_timeout - (deadline - time.time()))

                # Fail fast. A dead server produces the same message forever, and
                # waiting out the full timeout on a GPU costs real money.
                fatal = [
                    marker for marker in FATAL_LOG_MARKERS
                    if marker.lower() in tail.lower()
                ]
                if fatal or not self._server_alive():
                    detail = self._ssh(
                        "grep -iE 'error|failed|too old|not supported|Traceback' "
                        "/tmp/vllm.log | tail -20",
                        timeout=60,
                    ).stdout.strip()
                    raise RuntimeError(
                        "vllm failed to start "
                        f"({'log: ' + fatal[0] if fatal else 'process exited'}).\n"
                        + (detail or tail)
                    )
                _log(f"  still loading ({elapsed}s) {tail.splitlines()[-1][:90] if tail else ''}")
            time.sleep(10)
        raise TimeoutError(
            f"model did not come up within {self.serve_timeout:.0f}s; "
            f"check: ssh {self.alias} 'tail -50 /tmp/vllm.log'"
        )

    def save_server_log(self, path: str = "vllm-server.log") -> Optional[str]:
        """Copy the remote vllm log here. Call before teardown: the instance —
        and its disk, and therefore the only record of why it failed — is gone
        the moment it is terminated."""
        if not self.alias:
            return None
        result = self._ssh("cat /tmp/vllm.log", timeout=120)
        if result.returncode != 0 or not result.stdout:
            return None
        pathlib.Path(path).write_text(result.stdout)
        _log(f"server log saved to {path} ({len(result.stdout.splitlines())} lines)")
        return path

    def _teardown(self) -> None:
        if self._tunnel is not None:
            with contextlib.suppress(Exception):
                self._tunnel.terminate()
                self._tunnel.wait(timeout=10)
            self._tunnel = None

        if self.instance_id is None:
            if self._provisioned:
                self._sweep_orphans()
            return
        if self.keep_alive:
            _log(
                f"KEEPING {self.instance_id} alive as requested. It stops by itself "
                f"in <= {self.max_session_minutes} min; terminate sooner with:\n"
                f"  qbraid compute instances terminate {self.instance_id} --yes"
            )
            return

        _log(f"terminating {self.instance_id} ...")
        with contextlib.suppress(Exception):
            self.client.terminate_bma_instance(self.instance_id)
            _log("terminated")
        self.instance_id = None

    def _terminate_by_id(self, instance_id: str) -> None:
        """Backstop used by atexit; safe to call on an already-dead instance."""
        if self.keep_alive:
            return
        with contextlib.suppress(Exception):
            self.client.terminate_bma_instance(instance_id)

    def _sweep_orphans(self) -> None:
        """Kill anything that appeared since we started but we never addressed.

        Reached when provisioning succeeded and then something failed before we
        had an id to hold on to.
        """
        _log("sweeping for an instance we may have started ...")
        try:
            current = self.client.list_bma_instances()
        except Exception as exc:  # noqa: BLE001
            _log(f"could not list instances to sweep: {exc}")
            _log("CHECK MANUALLY: qbraid compute status")
            return

        for instance in current:
            identifier = _instance_id(instance)
            if not identifier or identifier in self._pre_existing:
                continue
            if getattr(instance, "profile_slug", None) != self.profile:
                continue
            _log(f"terminating orphan {identifier}")
            with contextlib.suppress(Exception):
                self.client.terminate_bma_instance(identifier)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--results-dir", default="results/remote_gpu")
    parser.add_argument("--max-session-minutes", type=int, default=120)
    parser.add_argument(
        "--local-port", type=int, default=LOCAL_PORT,
        help="Local end of the tunnel. Change it to run two GPUs at once.",
    )
    parser.add_argument(
        "--vllm-spec", default=DEFAULT_VLLM_SPEC,
        help="pip spec for vllm, e.g. 'vllm==0.8.5.post1' when the image's "
             "NVIDIA driver is older than the latest wheel expects.",
    )
    parser.add_argument(
        "--keep-alive", action="store_true",
        help="Do not terminate on exit. The hard session cap still applies.",
    )
    parser.add_argument(
        "--serve-only", action="store_true",
        help="Bring the endpoint up and hold it, so you can drive it yourself.",
    )
    args = parser.parse_args()

    endpoint = RemoteGPUEndpoint(
        profile=args.profile,
        model=args.model,
        max_session_minutes=args.max_session_minutes,
        local_port=args.local_port,
        keep_alive=args.keep_alive,
        vllm_spec=args.vllm_spec,
    )

    with endpoint as gpu:
        if args.serve_only:
            _log(f"serving at {gpu.base_url} -- Ctrl-C to tear down")
            with contextlib.suppress(KeyboardInterrupt):
                signal.pause()
            return 0

        return subprocess.run(
            [sys.executable, "run_evolution.py", "--endpoint", "local",
             "--base-url", gpu.base_url, "--model", gpu.model,
             "--generations", str(args.generations),
             "--results-dir", args.results_dir],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
