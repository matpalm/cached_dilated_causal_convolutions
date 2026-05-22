# cached dilated causal convolutions

a wavenet like architecture running on a daisy patch ( cortex-m7 ) at 48kHz and with a eurorack pmod and ecpix-5 FPGA at 192kHz

## running on tiliqua

* pretrain at FP 4.12, fine tune at FP 4.8
* trained on frequencies between A3 to A5

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        (None, 1024, 4)         0
 qconv_0_qb (QConv1D)        (None, 256, 4)           68
 qconv_1_qb (QConv1D)        (None, 64, 8)           136
 qconv_2_qb (QConv1D)        (None, 16, 8)           264
 qconv_3_qb (QConv1D)        (None, 4, 16)          528
 qconv_4_qb (QConv1D)        (None, 1, 1)           65
=================================================================
Total params: 1061 (4.14 KB)
```

```
Info:                 DP16KD:      27/     56    48%
Info:             MULT18X18D:      11/     28    39%
...
Info:             TRELLIS_FF:   12163/  24288    50%
Info:           TRELLIS_COMB:   14445/  24288    59%
```

just makes timing...

```
Info: Max frequency for clock '$glbnet$audio_clk': 76.27 MHz (PASS at 50.00 MHz)
Info: Max frequency for clock       '$glbnet$clk': 61.38 MHz (PASS at 60.00 MHz)
```

routing taking about 10mins

```
Info:     133296 |    39842      89385 |  125   171 |         0|      14.15     762.52|
```



note: my blog is borked currently :/ might try to fix it this weekend :/

see [the MCU blog post](http://matpalm.com/blog/wavenet_on_mcu/) followed by [the FPGA blog post](http://matpalm.com/blog/wavenet_on_fpga/)

![Wavenet running on a eurorack pmod and fpga](wavenet_fpga.png?raw=true)
