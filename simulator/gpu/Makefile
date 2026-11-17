# The engine library the Python backend loads (cns_metal.py). Fast math off
# inside the MSL (set in cns_engine.m); -O2 here only affects host code.
# The library carries the SHA-256 of its own two source files, and
# cns_metal.py refuses to load one whose stamp differs from the files on disk.
SRC_SHA256 := $(shell cat cns_engine.m cns_engine.h | shasum -a 256 | cut -c1-64)

libcnsengine.dylib: cns_engine.m cns_engine.h
	clang -O2 -fobjc-arc -dynamiclib -framework Metal -framework Foundation \
	    -DCNS_ENGINE_SRC_SHA256='"$(SRC_SHA256)"' \
	    -install_name @rpath/libcnsengine.dylib -o $@ cns_engine.m
