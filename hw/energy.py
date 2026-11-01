"""GPU (NVML) and CPU (RAPL, Linux only) energy measurement around a command.

GPU energy uses NVML's total-energy counter when the driver exposes it
(Volta and newer, including the RTX 4060), else integrates power samples.
'Above idle' subtracts idle power × duration, measured beforehand.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


class GPU:
    def __init__(self, index=0):
        self.ok = False
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml as nv
            nv.nvmlInit()
            self.nv = nv
            self.h = nv.nvmlDeviceGetHandleByIndex(index)
            self.name = nv.nvmlDeviceGetName(self.h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(self.h)
                self.counter = True
            except Exception:
                self.counter = False
            self.ok = True
        except Exception as e:
            self.err = str(e)

    def energy_mj(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None

    def sample(self):
        nv = self.nv
        s = {"t": time.time(), "power_w": nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0}
        for key, clk in (("sm_mhz", nv.NVML_CLOCK_SM), ("mem_mhz", nv.NVML_CLOCK_MEM)):
            try:
                s[key] = nv.nvmlDeviceGetClockInfo(self.h, clk)
            except Exception:
                pass
        try:
            s["temp_c"] = nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU)
            u = nv.nvmlDeviceGetUtilizationRates(self.h)
            s["util"] = u.gpu
        except Exception:
            pass
        return s


class RAPL:
    def __init__(self):
        self.paths = []
        base = Path("/sys/class/powercap")
        if base.exists():
            for d in sorted(base.glob("intel-rapl:*")):
                if d.name.count(":") == 1 and (d / "energy_uj").exists():
                    try:
                        int((d / "energy_uj").read_text())
                        self.paths.append(d)
                    except Exception:
                        pass
        self.ok = bool(self.paths)

    def read(self):
        return sum(int((p / "energy_uj").read_text()) for p in self.paths)

    def max_range(self):
        return sum(int((p / "max_energy_range_uj").read_text()) for p in self.paths)


class Meter:
    """with Meter() as m: ...  -> m.result"""

    def __init__(self, gpu_index=0, period=0.05):
        self.gpu = GPU(gpu_index)
        self.rapl = RAPL()
        self.period = period
        self.samples = []

    def _loop(self):
        while not self._stop.is_set():
            if self.gpu.ok:
                self.samples.append(self.gpu.sample())
            time.sleep(self.period)

    def __enter__(self):
        self._stop = threading.Event()
        self.t0 = time.time()
        self.e0 = self.gpu.energy_mj() if self.gpu.ok else None
        self.r0 = self.rapl.read() if self.rapl.ok else None
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join()
        t = time.time() - self.t0
        res = {"seconds": t}
        if self.gpu.ok:
            if self.e0 is not None:
                res["gpu_j"] = (self.gpu.energy_mj() - self.e0) / 1000.0
                res["gpu_j_source"] = "nvml_counter"
            elif len(self.samples) > 1:
                e = sum((b["t"] - a["t"]) * (a["power_w"] + b["power_w"]) / 2
                        for a, b in zip(self.samples, self.samples[1:]))
                res["gpu_j"] = e
                res["gpu_j_source"] = "nvml_samples"
            if self.samples:
                for k in ("power_w", "sm_mhz", "mem_mhz", "temp_c", "util"):
                    v = [s[k] for s in self.samples if k in s]
                    if v:
                        res[f"{k}_mean"] = sum(v) / len(v)
                        res[f"{k}_min"] = min(v)
                        res[f"{k}_max"] = max(v)
            res["gpu"] = self.gpu.name
        if self.rapl.ok:
            d = self.rapl.read() - self.r0
            if d < 0:
                d += self.rapl.max_range()
            res["cpu_j"] = d / 1e6
        self.result = res


def idle_power(seconds=10.0, gpu_index=0):
    with Meter(gpu_index) as m:
        time.sleep(seconds)
    r = m.result
    out = {"seconds": r["seconds"]}
    if "gpu_j" in r:
        out["gpu_w"] = r["gpu_j"] / r["seconds"]
    if "cpu_j" in r:
        out["cpu_w"] = r["cpu_j"] / r["seconds"]
    for k in ("sm_mhz_mean", "temp_c_mean"):
        if k in r:
            out[k] = r[k]
    return out


def run_metered(cmd, idle=None, gpu_index=0, cwd=None):
    with Meter(gpu_index) as m:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    r = dict(m.result)
    if idle:
        if "gpu_j" in r and "gpu_w" in idle:
            r["gpu_j_above_idle"] = r["gpu_j"] - idle["gpu_w"] * r["seconds"]
        if "cpu_j" in r and "cpu_w" in idle:
            r["cpu_j_above_idle"] = r["cpu_j"] - idle["cpu_w"] * r["seconds"]
    return p, r


def describe_env():
    g, r = GPU(), RAPL()
    return {"nvml": g.ok, "nvml_energy_counter": getattr(g, "counter", False),
            "gpu": getattr(g, "name", None), "rapl": r.ok, "platform": sys.platform,
            "pid": os.getpid()}
