# comfyui-zluda-gfx1030

> **This project has moved to [comfyui-zluda-rdna2](https://github.com/RYZENNAVI/comfyui-zluda-rdna2).**
> This repository is archived and no longer maintained. Nothing here is wrong, but everything here is also there, and the version there has been corrected in places this one was not.

## What happened

This project covered gfx1030 (RX 6950 XT, 6900 XT, 6800 XT, 6800). A sister project covered gfx1031 (RX 6700 XT, 6750 XT, 6700). Once both had been verified against real hardware and the findings were compared, almost everything turned out to be shared: the ZLUDA and HIP version matching, the DLLs that go into `torch\lib`, the launcher that overwrites the file you just patched, and the attention backend that resets the display driver.

Exactly one thing genuinely differs between the two architectures, and it is the kernels. gfx1030 is on the official AMD ROCm support list, so stock rocBLAS already ships its kernels. gfx1031 is not, so they have to be installed. The merged project detects which card you have and acts accordingly.

Keeping two repositories meant every fix had to be made twice, so they were merged.

## What the merged version fixes that this one got wrong

- **cuDNN.** This repository once claimed RDNA2 has no cuDNN engine under ZLUDA and that convolutions crash without it disabled. Measured on both architectures, that is not true: `cudnn.is_available()` returns true and convolutions run fine with it enabled.
- **Attention.** The driver resets blamed on cuDNN come from the mem-efficient SDPA backend, whose CUTLASS kernel is built for the wrong SM version. The merged docs name the real cause and the one line that guards against it.
- **Hardware claims.** The scripts accept every desktop RDNA2 card, but only an RX 6950 XT and an RX 6700 XT have actually been run on. The merged README says so instead of implying more.

## Go here instead

**https://github.com/RYZENNAVI/comfyui-zluda-rdna2**

The full history of this repository, including how the attention finding was tracked down, was merged into it rather than copied, so none of it was lost.
