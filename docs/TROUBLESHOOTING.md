# Troubleshooting

Look up what you are seeing. For most problems `scripts\Check-Environment.ps1` points straight at the cause.

## Error lookup

| Symptom | What is actually wrong | Fix |
|---|---|---|
| Exit code `0xC0000135`<br>(STATUS_DLL_NOT_FOUND) | The `amdhip64_N.dll` that ZLUDA wants is missing. ZLUDA's `nvcuda.dll` **hardcodes the major version** in its import table | Match them up: ZLUDA 3.9.5 needs HIP 6.x (`amdhip64_6.dll`), ZLUDA 3.9.6 needs HIP 7.x (`amdhip64_7.dll`) |
| Exit code `0xC0000139`<br>(STATUS_ENTRYPOINT_NOT_FOUND) | The DLL was found, but it does not export what was asked for | Still a version mix. Check whether a different version of `amdhip64*.dll` is being loaded first from PATH |
| `WinError 126: the specified module could not be found`<br>(on `import torch`) | Python 3.8+ **no longer searches PATH** for extension-module DLL dependencies | The DLLs must go directly into `venv\Lib\site-packages\torch\lib`. Editing PATH does nothing at all |
| `AttributeError: module 'comfy_kitchen' has no attribute 'int8_attention_is_available'` | That function does not exist under ZLUDA | `scripts\Patch-ComfyUI.ps1` |
| Access violation during convolution, or the process dies silently | RDNA2 has no cuDNN engine under ZLUDA | `scripts\Patch-ComfyUI.ps1`. If you patched `comfy\zluda.py` by hand, read "The launcher overwrites your patch" below |
| A fix you made to `comfy\zluda.py` has no effect | `comfyui.bat` regenerates that file on every launch | Edit `comfy\customzluda\zluda-default.py` instead |
| torch imports, but any CUDA call fails after replacing DLLs | `cublasLt64_11.dll` was overwritten with the ZLUDA `cublasLt.dll` | Only four DLLs get replaced. Reinstall torch to restore it |
| Access violation while loading a large model | The safetensors mmap path faults under memory pressure | Add `--disable-mmap` to the launch arguments |
| Out-of-memory part-way through a generation | Async offload or pinned memory | Launch with `--disable-async-offload --disable-pinned-memory` |
| The first generation hangs for ten minutes or more | Expected. ZLUDA is JIT-compiling kernels | Wait. They are cached in `%LOCALAPPDATA%\ZLUDA\ComputeCache` and later runs are fast |
| The screen goes black and recovers, event 4101 "display driver stopped responding", sometimes a hard hang | Something called torch SDPA, which resets the driver on this setup | Use split attention, for example `--use-quad-cross-attention`. See "torch SDPA resets the display driver" below |
| `no kernel image is available for execution` | Unexpected on gfx1030, which ships with kernels. The HIP install is stripped or broken | Reinstall the HIP SDK. Do not go looking for kernel packs; this card does not need them |
| HIP is installed but behaves as if it is not | Several versions installed with the wrong PATH order, or a user-scope environment variable shadowing the machine scope | See "Several HIP versions side by side" below |

## The awkward details

### The launcher overwrites your patch

`comfyui.bat` runs this twice during startup, before and after its git update:

```
copy comfy\customzluda\zluda-default.py comfy\zluda.py /y
```

`comfy\model_management.py` imports `comfy.zluda`, so that regenerated copy is the module that actually runs. Anything you edit in `comfy\zluda.py` survives until the next launch and no longer.

Note that `comfy\customzluda\zluda.py` is a third file. It is not the default, and it treats cuDNN differently: it reads `TORCH_BACKENDS_CUDNN_ENABLED` and **defaults to enabled**. If you switch to it, set that variable to `0`.

`zluda-default.py` also turns off `flash_sdp` and `mem_efficient_sdp` and forces `math_sdp` on. Leave that alone; the other SDPA backends have no working path here.

### Does cuDNN actually crash on gfx1030?

Not on the configuration this project was built against. On an RX 6950 XT with ZLUDA 3.9.5, HIP 6.4 and torch 2.7.0+cu118, `torch.backends.cudnn.is_available()` returns true, reports version 9.1.0, and fp16 convolutions run correctly with cuDNN **enabled** at tiny, VAE and UNet sizes alike.

That was re-measured with one operation per process on an otherwise idle machine, after an earlier round of testing had been confounded by other GPU work: 20 convolutions at 320->320 with cuDNN on took 0.019s each and left the driver alone. The driver resets seen during that earlier round came from attention, not convolution.

That is worth knowing, but it does not make the setting yours to choose: ComfyUI-Zluda disables cuDNN on import for every ZLUDA user, so that is how ComfyUI runs regardless. `Test-Setup.ps1` tests the configuration that ships, then reports the cuDNN-enabled result separately as information. If your card fails that informational line, say so in an issue.

Note that `torch\lib` keeps NVIDIA's own `cudnn64_9.dll` and friends. The ZLUDA `cudnn.dll` is not copied over them, the same way `cublasLt.dll` is not.

### torch SDPA resets the display driver

`torch.nn.functional.scaled_dot_product_attention` resets the display driver on this setup. The reset shows up as event 4101, a black screen that recovers, or a hang that needs the power button.

Measured on an RX 6950 XT with ZLUDA 3.9.5, HIP 6.4 and torch 2.7.0+cu118, one operation per process, machine otherwise idle:

| Operation | Result |
|---|---|
| `matmul` 1024x1024 fp16 | fine |
| `matmul` 256x256 fp16 | fine |
| `conv2d` 320->320 at 128x128, cuDNN off, 20x | fine |
| `conv2d` 320->320 at 128x128, **cuDNN on**, 20x | fine, 0.019s per call |
| **`SDPA` 1x8x4096x64 fp16, 10x** | **driver reset** |
| **`SDPA` 1x8x256x64 fp16, 10x** | **driver reset** |

Every run that called SDPA reset the driver; every run that did not was fine. Sequence length makes no difference, so this is not a kernel running past the `TdrDelay` timeout, which is 2 seconds by default: the small case finishes in milliseconds and still takes the driver down. Something in that code path is simply broken here.

This is what split attention is protecting you from. `--use-quad-cross-attention` computes attention as chunked matmuls and never enters the SDPA path, which is why a normally configured ComfyUI runs fine on this card while a bare `torch` script does not. **If you see 4101 during generation, check that flag first**, ahead of the model and your VRAM.

Raising `TdrDelay` does not help, because the timeout is not what is being hit.

`Test-Setup.ps1` therefore checks attention as chunked matmuls rather than calling SDPA. A self-test that resets your display driver is worse than no self-test.

If SDPA works on your gfx1030 card, please say so in an issue; it would be useful to know whether this is specific to one driver or ZLUDA build.

### HIP SDK installed, but `bin` has no `amdhip64.dll`

The installer sometimes skips the core runtime, leaving `bin` without the one DLL that matters.

Take it from the driver package:

```
C:\Windows\System32\DriverStore\FileRepository\amdocl.inf_amd64_*\
```

Several versions live there, and you **must pick the one matching your installed display driver**. The wrong version crashes just the same.

### Several HIP versions side by side

Three things have to line up, and any one of them being wrong breaks everything:

1. **PATH order** - the `bin` directory of the version you want has to come first.
2. **`HIP_PATH`** - note that a **user-scope variable overrides the machine scope**. The installer writes the machine scope, so a user-scope value you set by hand silently wins. This one is easy to miss.
3. **The ZLUDA major version** - see the error table above.

`Check-Environment.ps1` checks all three.

### Which DLLs go into torch\lib

Copy these four from the `zluda` directory, renaming as shown:

| Source | Destination in `venv\Lib\site-packages\torch\lib` |
|---|---|
| `cublas.dll` | `cublas64_11.dll` |
| `cusparse.dll` | `cusparse64_11.dll` |
| `cufft.dll` | `cufft64_10.dll` |
| `nvrtc.dll` | `nvrtc64_112_0.dll` |

Two things are deliberately **not** copied:

- **`cublasLt.dll`**. torch keeps its own `cublasLt64_11.dll`, which is a few hundred MB against ZLUDA's ~180 KB. Copying every DLL in the directory, which looks like the obvious thing to do, breaks torch.
- **`nvcuda.dll`**. `zluda.exe` injects it with Detours at launch; it does not belong in `torch\lib`.

To verify, compare hashes: each destination file should hash identically to its source.

### Killing a stuck self-test

Under ZLUDA, python regularly refuses to exit after GPU work, spinning on CUDA context teardown. `os._exit()` does not help, because what is stuck is the `zluda.exe` layer underneath.

`Test-Setup.ps1` handles this by printing a sentinel line when the work is done and tearing down the whole process tree once it appears, rather than waiting for the process to end on its own.

## Still stuck

Run these two and include their output in an issue:

```powershell
.\scripts\Check-Environment.ps1
.\scripts\Test-Setup.ps1
```
