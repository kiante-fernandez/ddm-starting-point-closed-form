# make [core|py|r] - build and self-check each layer
all: core py r

core:
	cd core && c++ -O2 -std=c++11 -DFDDM_TEST ddm_sz.cpp Faddeeva.cc -o check && ./check

py:
	cd python && cythonize -3 -i ddmsz.pyx && python check.py

r:
	cd R && rm -f *.o ../core/*.o && PKG_CPPFLAGS=-I../core R CMD SHLIB -o ddmsz.so ../core/ddm_sz.cpp ../core/Faddeeva.cc ddmsz_r.cpp && Rscript ddmsz.R

clean:
	rm -rf core/check core/*.o R/*.so R/*.o python/*.so python/ddmsz.cpp python/build

.PHONY: all core py r clean
