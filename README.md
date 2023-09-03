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
$ arm-none-eabi-g++ \
 --specs=nosys.specs cdcc_main.cpp \
  -I../libDaisy/Drivers/CMSIS/Include/ \
  -I../libDaisy/Drivers/CMSIS/DSP/Include \
  -DARM_MATH_CM7
```

```
cd cmsis_lib
cmake -Bbuild -D CMAKE_BUILD_TYPE=Release
cd build
make
./cdcc
```

