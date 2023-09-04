import numpy as np
import cmsisdsp as dsp

class Block(object):

    def __init__(self, c1_kernel, c1_bias, c2_kernel, c2_bias):
        assert len(c1_kernel.shape) == 3
        assert len(c1_bias.shape) == 1
        assert len(c2_kernel.shape) == 3
        assert len(c2_bias.shape) == 1
        self.kernel_size = c1_kernel.shape[0]
        self.in_d = c1_kernel.shape[1]
        self.c1_out_d = c1_kernel.shape[2]
        assert c1_bias.shape[0] == self.c1_out_d
        assert c2_kernel.shape[0] == 1
        assert c2_kernel.shape[1] == self.c1_out_d
        self.c2_out_d = c2_kernel.shape[2]
        assert c2_bias.shape[0] == self.c2_out_d
        self.c1_kernel = c1_kernel
        self.c1_bias = c1_bias
        self.c2_kernel = c2_kernel
        self.c2_bias = c2_bias

    def apply(self, x):
        # apply first Kx1 convolution
        c1_result = np.empty((1, self.c1_out_d), dtype=np.float32)
        c1_result = dsp.arm_fill_f32(0, self.c1_out_d)
        #return c1_result
        for k in range(self.kernel_size):
            x_mi = x[k:k+1,:]
            assert x_mi.shape == (1, self.in_d), f"x_mi={x_mi.shape} in_d={self.in_d}"
            kernel_mi = self.c1_kernel[k]
            assert kernel_mi.shape == (self.in_d, self.c1_out_d)
            _status, intermediate_result = dsp.arm_mat_mult_f32(x_mi, kernel_mi)
            c1_result = dsp.arm_add_f32(c1_result, intermediate_result)
        # add bias and apply RELU
        c1_result = dsp.arm_add_f32(c1_result, self.c1_bias)
        c1_result = np.maximum(c1_result, 0)

        # apply second 1x1 convolution
        x_mi = c1_result
        x_mi = x_mi.reshape((1, self.c1_out_d))
        #assert x_mi.shape == (1, OUT_D), x_mi.shape
        kernel_mi = self.c2_kernel[0]
        #assert kernel_mi.shape == (c1_out_d, c2_out_d), kernel_mi.shape
        _status, c2_result = dsp.arm_mat_mult_f32(x_mi, kernel_mi)
        # add bias and apply RELU
        c2_result = dsp.arm_add_f32(c2_result, self.c2_bias)
        c2_result = np.maximum(c2_result, 0)
        return c2_result
