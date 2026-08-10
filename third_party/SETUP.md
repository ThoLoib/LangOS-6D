# third_party/ — vendored external deps for Stage-3 (gitignored, reproducible)

These are **not** committed (large, external, reproducible). Recreate them on a
fresh checkout with the commands below. They are consumed by `object_retrieval/
stage3_metrics.py` (BOP-AR, D_sym) and `object_retrieval/stage3_render.py` (VSD
renderer), which put both dirs on `sys.path` automatically.

## bop_toolkit/  — BOP pose-error + AR (MSSD/MSPD/VSD)
```bash
git clone --depth 1 https://github.com/thodan/bop_toolkit.git \
    third_party/bop_toolkit
```

## pylibs/  — pip deps not in the oscar image (installed with --target so they
## persist on the bind-mounted repo without an image rebuild)
Run inside the oscar container:
```bash
pip install --target /app/third_party/pylibs \
    pypng imageio pyrender vispy PyOpenGL
```

## System GL libs for the VSD renderer (pyrender + EGL)
Not pip-installable — built into the image via `Dockerfile.egl` (a thin overlay
on `tholoi/oscar-plus`, because the main Dockerfile no longer rebuilds cleanly):
```bash
docker build -f Dockerfile.egl -t tholoi/oscar-plus .
```
plus `NVIDIA_DRIVER_CAPABILITIES=all` on the oscar service (docker-compose.yml)
so the NVIDIA driver injects libEGL_nvidia at runtime.
