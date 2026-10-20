"""A conservative allocator budget plus own-process GPU-memory watchdog."""
import os
import signal
import subprocess
import threading
import time


class GPUBudget:
    def __init__(self,device,limit_gb=20.,reserve_gb=2.):
        self.device=device
        self.limit=limit_gb*1e9
        self.reserve=reserve_gb*1e9
        self.stop_event=threading.Event()
        self.peak_bytes=0
        self.thread=None

    def memory(self):
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        amount=0
        for line in p.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)==2 and parts[0]==str(os.getpid()):
                try:amount+=float(parts[1])*1024**2
                except ValueError:raise RuntimeError('Cannot monitor own GPU memory on this server.')
        return amount

    def start(self):
        if self.device=='cpu':return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use gpace or explicitly choose STAGE_DEVICE=cpu (re-trains both stage targets).')
        self.memory()  # Fail before training when nvidia-smi is unavailable.
        free,total=torch.cuda.mem_get_info(0)
        allocatable=min(self.limit-self.reserve,free-self.reserve,total*.95)
        if allocatable<=0:
            raise RuntimeError('Not enough free GPU memory under the requested budget.')
        torch.cuda.set_per_process_memory_fraction(float(allocatable/total),0)
        print(f'[GPU BUDGET] allocator={allocatable/1e9:.2f} GB; process watchdog={self.limit/1e9:.2f} GB; decimal GB.',flush=True)
        def watch():
            failures=0
            while not self.stop_event.wait(.5):
                try:
                    used=self.memory();self.peak_bytes=max(self.peak_bytes,used);failures=0
                    if used>self.limit:
                        print('[GPU BUDGET EXCEEDED] Stopping only this training process; completed caches remain.',flush=True)
                        os.kill(os.getpid(),signal.SIGTERM)
                        return
                except Exception as exc:
                    failures+=1
                    if failures>=3:
                        print(f'[GPU MONITOR FAILED] {exc}; stopping this process.',flush=True)
                        os.kill(os.getpid(),signal.SIGTERM)
                        return
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=12)
