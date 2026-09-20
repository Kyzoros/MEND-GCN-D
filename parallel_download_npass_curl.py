"""Parallel chunked download of NPASS3.0_activities.txt using curl.exe (proven to work
with -k) + HTTP Range. One curl process per chunk, run concurrently; concatenate at end.
Beats per-connection bandwidth cap if the server enforces one."""
import os, sys, subprocess, time

URL = "https://bidd.group/NPASS/downloadFiles/NPASS3.0_activities.txt"
OUT = "data/raw/NPASS/NPASS3.0_activities.txt"
CHUNKDIR = "data/raw/NPASS/_chunks"
TOTAL = 105252592
N = 16

os.makedirs(CHUNKDIR, exist_ok=True)
seg = TOTAL // N
parts = []
for i in range(N):
    s = i * seg
    e = TOTAL - 1 if i == N - 1 else i * seg + seg - 1
    parts.append((i, s, e))

t0 = time.time()
print(f"launching {N} parallel curls (total {TOTAL/1e6:.1f} MB)...")
procs = []
for i, s, e in parts:
    path = os.path.join(CHUNKDIR, f"chunk_{i}.bin")
    procs.append((i, s, e, subprocess.Popen(
        ["curl.exe", "-k", "-s", "-r", f"{s}-{e}", "-o", path, URL])))
ok = True
for i, s, e, p in procs:
    rc = p.wait()
    path = os.path.join(CHUNKDIR, f"chunk_{i}.bin")
    exp = e - s + 1
    got = os.path.getsize(path) if os.path.exists(path) else 0
    status = "OK" if (rc == 0 and got == exp) else f"FAIL rc={rc} got={got}/{exp}"
    if rc != 0 or got != exp:
        ok = False
    print(f"  chunk {i}: {status}")
dt = time.time() - t0
print(f"download phase: {dt:.1f}s, all_ok={ok}")

if not ok:
    print("SOME CHUNKS FAILED -> abort, do not overwrite good file")
    sys.exit(1)

with open(OUT, "wb") as out:
    for i, s, e in parts:
        with open(os.path.join(CHUNKDIR, f"chunk_{i}.bin"), "rb") as c:
            out.write(c.read())
        os.remove(os.path.join(CHUNKDIR, f"chunk_{i}.bin"))
sz = os.path.getsize(OUT)
print(f"concatenated -> {OUT}: {sz:,} bytes (expected {TOTAL:,}) match={sz==TOTAL}")
