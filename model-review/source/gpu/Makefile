CC      ?= cc
NVCC    ?= nvcc
CFLAGS  = -O3 -march=native -Wall -Wextra -std=c11
OMP     = -fopenmp

bench-ref: bench.c cns_ref.c cns_core.h
	$(CC) $(CFLAGS) -o $@ bench.c cns_ref.c -lm

bench-ref-omp: bench.c cns_ref.c cns_core.h
	$(CC) $(CFLAGS) $(OMP) -o $@ bench.c cns_ref.c -lm

bench-cuda: bench.c cns_ref.c cns_cuda.cu cns_core.h
	$(NVCC) -O3 -arch=native -DHAVE_CUDA -o $@ bench.c cns_ref.c cns_cuda.cu -lm

bench-metal: cns_metal.m
	clang -O2 -fobjc-arc -framework Metal -framework Foundation -o $@ cns_metal.m

clean:
	rm -f bench-ref bench-ref-omp bench-cuda bench-metal

.PHONY: clean

# The engine library the Python backend loads (cns_metal.py). Fast math off
# inside the MSL (set in cns_engine.m); -O2 here only affects host code.
# The library carries the SHA-256 of its own two source files, and
# cns_metal.py refuses to load one whose stamp differs from the files on disk
# (a stale dylib ran the network silent with no error, 2026-09-02).
SRC_SHA256 := $(shell cat cns_engine.m cns_engine.h | shasum -a 256 | cut -c1-64)

libcnsengine.dylib: cns_engine.m cns_engine.h
	clang -O2 -fobjc-arc -dynamiclib -framework Metal -framework Foundation \
	    -DCNS_ENGINE_SRC_SHA256='"$(SRC_SHA256)"' \
	    -install_name @rpath/libcnsengine.dylib -o $@ cns_engine.m
