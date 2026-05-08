changes compared to verilog_version

* all stream based; FP4.12 for all the neural net math
* dynamic sizing based on qkeras weights
* Memory or Array (EBR) based activation caches
* pipelining for the elements of the multiply

```
Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        (None, 256,  4)          0
 qconv_0_qb (QConv1D)        (None,  64,  4)          68
 qconv_1_qb (QConv1D)        (None,  16,  8)          136
 qconv_2_qb (QConv1D)        (None,   4, 16)          528
 qconv_3_qb (QConv1D)        (None,   1,  4)          260
=================================================================
Total params: 992
```

```
Info: 	              DP16KD:      36/     56    64%    # conv 1 and 2 activation caches
Info: 	          MULT18X18D:       5/     28    17%
...
Info: 	          TRELLIS_FF:    6306/  24288    25%
Info: 	        TRELLIS_COMB:   10620/  24288    43%
```

TODO:

* next botteneck is storing the activation cache for another layer depth; need to use PSRAM
* fp35 branch has WIP with FP2.14 pretraining -> FP3.5 fine tuning
* first mu_law test didn't do well ( but see there was a bug )
* need to retry the po2 quantisation and logic gate nets