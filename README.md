# 3d-string-art
Making 3D string art.

NOTE: must run on (Linux OR WSL with X11 forwarding) with open3d 0.15.2 to support offscreen rendering.

If you get a GLXBadFBConfig error like https://github.com/isl-org/Open3D/issues/2836, add this to .profile:
```
export LIBGL_ALWAYS_INDIRECT=0
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
export LIBGL_ALWAYS_SOFTWARE=1
```

If you get an error about `swrast` or `iris`, add this to .bashrc:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```