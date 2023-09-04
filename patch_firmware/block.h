#pragma once

inline void relu(float* a, size_t n) {
  while (n>0) {
    if (*a < 0) {
      *a = 0;
    }
    a++;
    n--;
  }
}

class Block {
  public:
    Block(size_t kernel_size, size_t in_d, size_t out_d,
          float* c1_kernel, float* c1_bias,
          float* c2_kernel, float* c2_bias
          ) :
            kernel_size_(kernel_size), in_d_(in_d), out_d_(out_d),
            c1_kernel_(c1_kernel), c1_bias_(c1_bias),
            c2_kernel_(c2_kernel), c2_bias_(c2_bias)
            {}

    void Apply(float* x,         // ( kernel_size_, in_dim)
               float* result) {  // ( out_dim, )

      //arm_status status;

      // zero results
      arm_fill_f32(0, result, out_d_);

      // run first convolution...

      // setup matrix instances for each of the K mat muls
      // including an instance for the result
      arm_matrix_instance_f32 x_mi;
      arm_mat_init_f32(&x_mi, 1, in_d_, x);
      arm_matrix_instance_f32 kernel_mi;
      arm_mat_init_f32(&kernel_mi, in_d_, out_d_, c1_kernel_);
      float matmul_result[out_d_];
      arm_matrix_instance_f32 matmul_result_mi;
      arm_mat_init_f32(&matmul_result_mi, 1, out_d_, matmul_result);

      // run the mat muls for each of the kernel steps
      for (size_t k=0; k < kernel_size_; k++) {
        // point matrix instances at next kernel
        x_mi.pData = (float *)x + (k * in_d_);
        kernel_mi.pData = (float *)c1_kernel_ + (k * in_d_ * out_d_);
        // do the mat mul
        arm_mat_mult_f32(&x_mi, &kernel_mi, &matmul_result_mi);
        // accumulate into result for c1
        arm_add_f32(result, matmul_result, result, out_d_);
      }

      // apply bias ( in place on result) and relu
      arm_add_f32(result, c1_bias_, result, out_d_);
      relu(result, out_d_);

      // run second convolution....

      // since second convolution is 1x1 it only needs to be
      // run once. it can also use the same matmul result from
      // conv1 since the two convolutions share the same number of
      // filters.
      // TODO: init for c2_kernel_ mi is the same every apply
      //         =>  could be cached
      arm_matrix_instance_f32 c1_out_mi;
      arm_mat_init_f32(&c1_out_mi, 1, out_d_, result);
      arm_mat_init_f32(&kernel_mi, out_d_, out_d_, c2_kernel_);
      arm_mat_mult_f32(&c1_out_mi, &kernel_mi, &matmul_result_mi);
      // apply bias (this time from matmul_result to result) and relu
      arm_add_f32(matmul_result, c2_bias_, result, out_d_);
      relu(result, out_d_);
    }

    const size_t kernel_size_;
    const size_t in_d_;
    const size_t out_d_;
    float* c1_kernel_;  // (kernel_size_, in_d, out_d)
    float* c1_bias_;    // (out_d,)
    float* c2_kernel_;  // (1, out_d, out_d)
    float* c2_bias_;    // (out_d,)
};