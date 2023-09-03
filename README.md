# cached dilated causal convolutions

1D dilated causal convolutions with extreme caching. intended for fast inference on the daisy platform.
trades off compute for heavy SDRAM caching of intermediate results.

`libdaisy` included as a submodule and contains the CMSIS lib intended to run on the daisy

## prototype notebooks

is the dumping ground for prototyping notebooks

uses https://pypi.org/project/cmsisdsp/

`conda activate cdcc`

## cmsis lib

`cmsis_lib/` contains the library code for running on the daisy

```
cd cmsis_lib
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
cd build
make
./cdcc
```

