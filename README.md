# PyNATS
Python network analysis for time series

For installing:
- You need llvmlite, which requires LLVM 10.0.x or 9.0.x and should be fine for most systems however in arch-linux this needs to be installed via the AUR (python-llvmlite) or by installing llvm10 using pacman (which will overwrite the latest version)
- Same for PyTorch (install with python-pytorch). If you cannot, then torch might be able to be installed via pip, however I would use the --no-cache-dir flag otherwise there's a MemoryError raised.
- You need cairo install (for pycairo -- sorry, this one's a pain.)
