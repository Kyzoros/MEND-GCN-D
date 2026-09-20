"""Parallel chunked download of NPASS3.0_activities.txt using HTTP Range requests.
Server confirmed to support Range (HTTP 206). Splits into N chunks downloaded
concurrently -> beats per-connection bandwidth cap if the server enforces one."""
import os, ssl, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://bidd.group/NPASS/downloadFiles/NPASS3.0_activities.txt"
OUT = "data/raw/NPASS/NPASS3.0_activities.txt"
N = 16

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get_total():
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        cr = r.headers.get("Content-Range")
        return int(cr.split("/")[1])

def dl_chunk(p):
    s, e = p
    req = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
    data = urllib.request.urlopen(req, context=ctx, timeout=180).read()
    with open(OUT, "r+b") as f:
        f.seek(s)
        f.write(data)
    return len(data)

def main():
    total = get_total()
    print(f"total size = {total:,} bytes ({total/1e6:.1f} MB), splitting into {N} chunks")
    seg = total // N
    parts = [(i * seg, total - 1 if i == N - 1 else i * seg + seg - 1) for i in range(N)]
    with open(OUT, "wb") as f:
        f.truncate(total)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(dl_chunk, p) for p in parts]
        for fut in futs:
            done += fut.result()
    dt = time.time() - t0
    print(f"DONE {done:,} bytes in {dt:.1f}s  -> {done/1e6/dt:.2f} MB/s")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
