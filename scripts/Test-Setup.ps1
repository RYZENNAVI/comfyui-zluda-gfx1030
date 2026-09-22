<#
.SYNOPSIS
Run real work on the GPU from the ComfyUI venv to prove the setup works.

.DESCRIPTION
Check-Environment.ps1 only inspects files and environment variables. This one
actually dispatches to the GPU:
  - matmul        goes through rocBLAS
  - convolution   fails here when cuDNN was not properly disabled
  - attention     goes through SDPA

The first run is slow because ZLUDA JIT-compiles kernels, cached under
%LOCALAPPDATA%\ZLUDA\ComputeCache. Later runs are fast.

Refuses to start while ComfyUI is running, so the two do not fight over VRAM.

.PARAMETER ComfyUIRoot
ComfyUI root directory. Auto-detected when omitted.

.PARAMETER TimeoutSeconds
How long to wait for a result. Defaults to 900, since the first run has to
JIT-compile kernels.

.PARAMETER Force
Run even when ComfyUI appears to be running.
#>
[CmdletBinding()]
param(
    [string]$ComfyUIRoot,
    [int]$TimeoutSeconds = 900,
    [switch]$Force
)

. (Join-Path $PSScriptRoot 'Common.ps1')

$root = Find-ComfyUIRoot -Hint $ComfyUIRoot
if (-not $root) { throw "ComfyUI root not found. Pass -ComfyUIRoot." }

$py = Join-Path $root 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "No $py - the ComfyUI venv is not set up." }

$zdir = Find-ZludaDir -ComfyUIRoot $root
if (-not $zdir) { throw "The ZLUDA directory was not found under $root." }
$zexe = Join-Path $zdir 'zluda.exe'
if (-not (Test-Path $zexe)) { throw "No zluda.exe in $zdir." }

# Sharing the GPU with a running ComfyUI gives confusing out-of-memory failures.
# Report it and stop; never kill a process this script did not start.
$busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, 'OrdinalIgnoreCase') })
if ($busy.Count -gt 0 -and -not $Force) {
    Write-Host "ComfyUI looks like it is running (PID $($busy[0].ProcessId))." -ForegroundColor Yellow
    Write-Host "Close it first so the two do not fight over VRAM, or pass -Force." -ForegroundColor Yellow
    exit 1
}

$code = @'
import torch, sys, os

print("torch       ", torch.__version__)
print("cuda avail  ", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU, stopping here.")
print("device      ", torch.cuda.get_device_name(0))

# ComfyUI runs with cuDNN off, because comfy\zluda.py turns it off on import.
# A bare interpreter does not, so set it here or this would test a different
# configuration from the one that matters.
torch.backends.cudnn.enabled = False
print("cudnn       ", torch.backends.cudnn.enabled, "(forced off, the way ComfyUI runs)")
print()

fail = []

# rocBLAS
try:
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    r = (a @ a).float().sum().item()
    assert r == r, "matmul produced NaN"
    print("matmul fp16  OK")
except Exception as e:
    fail.append(("matmul fp16", e))
    print("matmul fp16  FAILED:", e)

# convolution: blows up here when cuDNN was not disabled
try:
    x = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
    w = torch.randn(8, 4, 3, 3, device="cuda", dtype=torch.float16)
    y = torch.nn.functional.conv2d(x, w, padding=1)
    assert y.shape == (1, 8, 64, 64), y.shape
    print("conv2d fp16  OK")
except Exception as e:
    fail.append(("conv2d fp16", e))
    print("conv2d fp16  FAILED:", e)

# attention. Keep this small on purpose. A single SDPA call over a long sequence
# runs for well over a second on this card, and Windows resets the display driver
# at 2 seconds by default. 1x8x4096x64 reliably triggers that reset; this does not.
try:
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print("sdpa fp16    OK")
except Exception as e:
    fail.append(("sdpa fp16", e))
    print("sdpa fp16    FAILED:", e)

# Informational. ComfyUI-Zluda disables cuDNN for everyone, on the grounds that
# RDNA2 has no working engine under ZLUDA. On an RX 6950 XT with ZLUDA 3.9.5,
# HIP 6.4 and torch 2.7.0+cu118 this passes anyway. Either result is fine; it is
# reported so people can say what their own card does.
try:
    torch.backends.cudnn.enabled = True
    x = torch.randn(1, 320, 128, 128, device="cuda", dtype=torch.float16)
    w = torch.randn(320, 320, 3, 3, device="cuda", dtype=torch.float16)
    torch.nn.functional.conv2d(x, w, padding=1)
    torch.cuda.synchronize()
    print("conv2d cudnn on   works here (informational, not required)")
except Exception as e:
    print("conv2d cudnn on   fails here (informational, expected):", e)
finally:
    torch.backends.cudnn.enabled = False

print()
rc = 0
if fail:
    print("%d of 3 checks failed." % len(fail))
    rc = 1
else:
    print("All checks passed; this GPU can run ComfyUI.")

print("__GFX1030_DONE__ %d" % rc)
sys.stdout.flush()
os._exit(rc)
'@

$tmp = Join-Path $env:TEMP "gfx1030-selftest-$PID.py"
$log = Join-Path $env:TEMP "gfx1030-selftest-$PID.log"
Set-Content -Path $tmp -Value $code -Encoding UTF8

$env:PYTHONIOENCODING = 'utf-8'

Write-Host "Running the self-test in $root (the first run is slow while ZLUDA JIT-compiles kernels)..."
Write-Host ""

$p = Start-Process -FilePath $zexe -ArgumentList @('--', $py, $tmp) `
    -WorkingDirectory $root -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

# Once the work is done, python under zluda.exe often refuses to exit, spinning
# on CUDA context teardown. So wait for the sentinel line rather than for exit.
$rc = $null
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if (Test-Path $log) {
        $m = Select-String -Path $log -Pattern '^__GFX1030_DONE__ (\d+)' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($m) { $rc = [int]$m.Matches[0].Groups[1].Value; break }
    }
    if ($p.HasExited) { $rc = $p.ExitCode; break }
}

# The sentinel is printed immediately before os._exit, so give the process a
# moment to go on its own. Killing zluda.exe while it is still tearing down a
# CUDA context can take the display driver with it.
for ($i = 0; $i -lt 20 -and -not $p.HasExited; $i++) { Start-Sleep -Milliseconds 500 }
if (-not $p.HasExited) { Stop-ProcessTree -Id $p.Id }

if (Test-Path $log) {
    Get-Content $log -Encoding UTF8 | Where-Object { $_ -notmatch '^__GFX1030_DONE__' } | ForEach-Object { Write-Host $_ }
}
if ((Test-Path "$log.err") -and (Get-Item "$log.err").Length -gt 0) {
    Write-Host "--- stderr ---" -ForegroundColor DarkGray
    Get-Content "$log.err" -Encoding UTF8 | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray }
}
Remove-Item $tmp, $log, "$log.err" -ErrorAction SilentlyContinue

Write-Host ""
if ($rc -eq 0) {
    Write-Host "Self-test passed." -ForegroundColor Green
} elseif ($null -eq $rc) {
    Write-Host "No result after $TimeoutSeconds seconds. A first run really can take longer while kernels compile; raise -TimeoutSeconds and try again." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "Self-test failed. Look the errors above up in docs\TROUBLESHOOTING.md." -ForegroundColor Red
    exit 1
}
