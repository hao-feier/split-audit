#!/usr/bin/python3
"""Dependency-free memory watcher for exp1_composite.py.

Samples the exp1_composite.py process tree once per second using only /proc
(no psutil). Writes timestamped RSS/Swap per PID to stdout. Exits a few seconds
after the target process vanishes, recording the disappearance moment.
"""
import os
import sys
import time
import glob

TARGET = "exp1_composite.py"
WATCH_AFTER_DEATH_S = 20.0
SLEEP_S = 1.0
DELAY_MS = int(os.environ.get("WATCH_DELAY_MS", "0"))


def proc_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def rss_swap(pid):
    rss = swap = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])  # kB
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])  # kB
    except OSError:
        rss = swap = -1
    return rss, swap


def target_pids():
    out = []
    for pid in glob.glob("/proc/[0-9]*"):
        pid = int(pid.split("/")[2])
        if TARGET in proc_cmdline(pid):
            out.append(pid)
    return out


def ancestors_include_target(pid):
    seen = 0
    cur = pid
    while cur > 1 and seen < 32:
        try:
            with open(f"/proc/{cur}/stat") as f:
                parts = f.read().rsplit(")", 1)
                if len(parts) != 2:
                    return False
                ppid = int(parts[1].split()[1])
        except OSError:
            return False
        if TARGET in proc_cmdline(cur):
            return True
        cur = ppid
        seen += 1
    return False


def main():
    if DELAY_MS:
        time.sleep(DELAY_MS / 1000.0)
    # Wait up to 60 s for the process to appear (in case shell must start it).
    deadline = time.time() + 60.0
    while time.time() < deadline:
        pids = target_pids()
        if pids:
            break
        time.sleep(0.25)

    now = time.time()
    print(f"# watcher start {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}",
          flush=True)
    print(f"# system RSS_KB rss_total_kb swap_total_kb", flush=True)
    total_rss_total = total_swap_total = 0.0
    mem_total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                    break
    except OSError:
        pass

    alive = True
    quiet = 0
    while True:
        pids = target_pids()
        if not pids:
            alive = False
            quiet += 1
            if quiet * SLEEP_S >= WATCH_AFTER_DEATH_S:
                break
        else:
            quiet = 0
        rows = []
        rss_sum = swap_sum = 0
        # Only watch the target + its direct/indirect children.
        for pid in pids:
            rss, swap = rss_swap(pid)
            if rss < 0:
                continue
            rss_sum += rss
            swap_sum += swap
            rows.append(f"{pid}:{rss}kB/{swap}kB")
        # Children with target in ancestry (usually just the one process).
        extra = []
        for pid in glob.glob("/proc/[0-9]*"):
            pid = int(pid.split("/")[2])
            if pid in pids:
                continue
            if ancestors_include_target(pid):
                rss, swap = rss_swap(pid)
                if rss < 0:
                    continue
                rss_sum += rss
                swap_sum += swap
                extra.append(f"{pid}:{rss}kB/{swap}kB")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        frac = ""
        if mem_total:
            frac = " mem_%0.1f%%" % (100.0 * rss_sum / mem_total)
        line = f"{ts} rss={rss_sum/1024/1024:.2f}GiB swap={swap_sum/1024/1024:.2f}GiB" + frac
        if rows or extra:
            line += " [" + " ".join(rows) + "]" + ((" +children[" + " ".join(extra) + "]") if extra else "")
        else:
            line += " (no target)"
        print(line, flush=True)
        time.sleep(SLEEP_S)

    end = time.time()
    print(f"# watcher end {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))} "
          f"target_gone_since={quiet*SLEEP_S:.0f}s last_alive={alive}", flush=True)


if __name__ == "__main__":
    main()
