"""Lightweight Windows process/AC monitoring; never kills a process.

Uses Windows counters and NVML observations, not an extra GPU workload.
Raw activity logs are retained locally alongside every exclusive job.
"""
import ctypes
from ctypes import wintypes as wt
import json
import os
from pathlib import Path
import threading
import time


class ProcessEntry(ctypes.Structure):
    _fields_=[('size',wt.DWORD),('usage',wt.DWORD),('pid',wt.DWORD),('heap',ctypes.c_size_t),
              ('module',wt.DWORD),('threads',wt.DWORD),('ppid',wt.DWORD),('priority',wt.LONG),
              ('flags',wt.DWORD),('name',wt.WCHAR*260)]


class PowerStatus(ctypes.Structure):
    _fields_=[('ac',wt.BYTE),('flags',wt.BYTE),('percent',wt.BYTE),('system',wt.BYTE),
              ('remaining',wt.DWORD),('full',wt.DWORD)]


def snapshot():
    if os.name!='nt':raise RuntimeError('This testbed monitor targets the agreed Windows host')
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes=[wt.DWORD,wt.DWORD];k.CreateToolhelp32Snapshot.restype=wt.HANDLE
    for name in ('Process32FirstW','Process32NextW'):
        f=getattr(k,name);f.argtypes=[wt.HANDLE,ctypes.POINTER(ProcessEntry)];f.restype=wt.BOOL
    k.OpenProcess.argtypes=[wt.DWORD,wt.BOOL,wt.DWORD];k.OpenProcess.restype=wt.HANDLE
    k.CloseHandle.argtypes=[wt.HANDLE];k.CloseHandle.restype=wt.BOOL
    k.GetProcessTimes.argtypes=[wt.HANDLE,*([ctypes.POINTER(wt.FILETIME)]*4)];k.GetProcessTimes.restype=wt.BOOL
    stamp=lambda f:((f.dwHighDateTime<<32)+f.dwLowDateTime)*1e-7
    h=k.CreateToolhelp32Snapshot(2,0)
    if h==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    processes=[]
    try:
        entry=ProcessEntry();entry.size=ctypes.sizeof(entry)
        ok=k.Process32FirstW(h,ctypes.byref(entry))
        while ok:
            p={'pid':entry.pid,'ppid':entry.ppid,'name':entry.name,'cpu_seconds':None}
            handle=k.OpenProcess(0x1000,False,entry.pid)
            if handle:
                try:
                    created,exited,kernel,user=(wt.FILETIME() for _ in range(4))
                    if k.GetProcessTimes(handle,*map(ctypes.byref,(created,exited,kernel,user))):
                        p['cpu_seconds']=stamp(kernel)+stamp(user)
                finally:k.CloseHandle(handle)
            processes.append(p);ok=k.Process32NextW(h,ctypes.byref(entry))
    finally:k.CloseHandle(h)
    power=PowerStatus();available=k.GetSystemPowerStatus(ctypes.byref(power))
    return {'unix':time.time(),'processes':processes,'ac_line':power.ac if available else None,
            'logical_cpus':os.cpu_count()}


def descendants(catalog,root):
    ids={root}
    while True:
        expanded=ids|{p['pid'] for p in catalog.values() if p['ppid'] in ids}
        if expanded==ids:return ids
        ids=expanded


def external_flags(samples,root,*,cpu_threshold=5.0,consecutive=3):
    """Condition-based flags, independent of speed or game outcome."""
    catalog={p['pid']:p for s in samples for p in s['processes']}
    owned=descendants(catalog,root)
    permitted={'codex.exe','dwm.exe','explorer.exe','system idle process'}
    close_first={'chrome.exe','msedge.exe','ms-teams.exe','discord.exe','obs64.exe','vlc.exe',
                 'steam.exe','docker desktop.exe','vmmem.exe','vmmemwsl.exe','vmware-vmx.exe','virtualboxvm.exe'}
    flags=set();streak={}
    if any(s.get('ac_line')==0 for s in samples):flags.add('AC power disconnected')
    for s in samples:
        for p in s['processes']:
            if p['pid'] not in owned and p['name'].lower() in close_first:
                flags.add('close nonessential application: '+p['name'])
        for pid in s.get('gpu_compute_pids') or []:
            if pid not in owned and catalog.get(pid,{}).get('name','').lower() not in permitted:
                flags.add('external GPU compute PID: '+str(pid))
    for before,after in zip(samples,samples[1:]):
        delta=after['unix']-before['unix']
        if delta<=0:continue
        old={p['pid']:p for p in before['processes']}
        for p in after['processes']:
            pid=p['pid']
            if pid in owned or p['name'].lower() in permitted:continue
            previous=old.get(pid,{}).get('cpu_seconds');current=p.get('cpu_seconds')
            if previous is None or current is None:continue
            pct=100*max(current-previous,0)/(delta*after['logical_cpus'])
            streak[pid]=streak.get(pid,0)+1 if pct>cpu_threshold else 0
            if streak[pid]>=consecutive:flags.add('sustained external CPU activity: '+p['name'])
    return sorted(flags)


class ActivityMonitor:
    def __init__(self,path,period=2):
        self.path=Path(path);self.period=period;self.samples=[];self.errors=[]
        self.stop=threading.Event();self.gpu=None

    def sample(self):
        try:
            s=snapshot()
            if self.gpu and self.gpu.ok:
                s['gpu']=self.gpu.sample()
                try:
                    rows=self.gpu.nv.nvmlDeviceGetComputeRunningProcesses(self.gpu.h)
                    s['gpu_compute_pids']=[r.pid for r in rows]
                except Exception:s['gpu_compute_pids']=None
            self.samples.append(s)
            self.stream.write(json.dumps(s)+'\n');self.stream.flush()
        except Exception as exc:
            self.errors.append(repr(exc));self.stream.write(json.dumps({'monitor_error':repr(exc)})+'\n')

    def loop(self):
        while not self.stop.wait(self.period):self.sample()

    def __enter__(self):
        from hw.energy import GPU
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('x',encoding='utf-8');self.gpu=GPU()
        if not self.gpu.ok:self.errors.append('NVML monitoring unavailable: '+str(getattr(self.gpu,'err','unknown')))
        self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample();self.stream.close()

    def report(self,root):
        return {'samples':len(self.samples),'period_seconds':self.period,
                'flags':external_flags(self.samples,root),'monitor_errors':self.errors,
                'gpu_process_visibility':'NVML compute process query; WDDM graphics activity may not be fully visible',
                'rules':'5% total-machine CPU for 3 consecutive samples; nonessential app presence; external compute PID; AC disconnect',
                'raw_log':str(self.path)}
