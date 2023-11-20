the "top" modules are `qb_network` and `po2_network`

`qb_network` represents the quantised bits trained version and uses a normal multiply for the `dot_product.sv`.

`po2_` represents a power-of-two quantised trained version that uses some of the same utils as the `qb_network` version,
e.g. `activation_cache` and `left_shift_buffer`
but has a different version of the dot product ( `po2_dot_product` ) that instead of normal multiplies implements everything wuth shift operations ( represented by the `po2_multiply` module )

see also https://github.com/matpalm/eurorack-pmod/blob/master/gateware/cores/
for the slightly different versions running on the actual eurorack-pmod / ecpix5 combination

