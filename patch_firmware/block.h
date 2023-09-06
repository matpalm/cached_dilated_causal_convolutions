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
            c1_kernel_(c1_kernel),
            c1_bias_(c1_bias), c2_bias_(c2_bias) {

      // prepare a matrix instance for x
      // for now set it to what, will become, invalid memory but ignore
      // since we'll reset the pData when required
      float i_too_like_to_live_dangerously[in_d] = {0};
      arm_mat_init_f32(&x_mi_, 1, in_d, i_too_like_to_live_dangerously);

      // prep matrix instances for kernels
      arm_mat_init_f32(&c1_kernel_mi_, in_d, out_d, c1_kernel);
      arm_mat_init_f32(&c2_kernel_mi_, out_d, out_d, c2_kernel);

      // prepare matrix for matmul results, and allocate it's memory
      // against instance variable
      matmul_result_ = new float[out_d];
      arm_mat_init_f32(&matmul_result_mi_, 1, out_d, matmul_result_);

      // allocate buffer for input
      input_buffer_ = new float[kernel_size * in_d];
    }

    float* GetInputBuffer() { return input_buffer_; }
    const size_t GetInputBufferSize() { return kernel_size_ * in_d_; }
    void SetOutputBuffer(float* output_buffer) { output_buffer_ = output_buffer; }
    const size_t GetOutputBufferSize() { return out_d_; }

    void Run() { // ( out_dim, )
      // expects value to be ready in input_buffer_

      // zero results
      arm_fill_f32(0, output_buffer_, out_d_);

      // run first convolution...

      // run the mat muls for each of the kernel steps
      // recall; the matrix instances for x and c1_kernel has already
      // been setup, then just need their pData shifted along for each kernel
      for (size_t k=0; k < kernel_size_; k++) {
        // point matrix instances at next kernel
        x_mi_.pData = (float *)input_buffer_ + (k * in_d_);
        c1_kernel_mi_.pData = (float *)c1_kernel_ + (k * in_d_ * out_d_);
        // do the mat mul
        arm_mat_mult_f32(&x_mi_, &c1_kernel_mi_, &matmul_result_mi_);
        // accumulate into result for c1
        arm_add_f32(output_buffer_, matmul_result_, output_buffer_, out_d_);
      }

      // apply bias ( in place on result ) and relu
      arm_add_f32(output_buffer_, c1_bias_, output_buffer_, out_d_);
      relu(output_buffer_, out_d_);

      // run second convolution....

      // since second convolution is 1x1 it only needs one mat mul.
      // it can also use the same matmul result from
      // conv1 since the two convolutions share the output feature depth.
      arm_matrix_instance_f32 c1_out_mi;
      arm_mat_init_f32(&c1_out_mi, 1, out_d_, output_buffer_);
      arm_mat_mult_f32(&c1_out_mi, &c2_kernel_mi_, &matmul_result_mi_);
      // apply bias (this time from matmul_result to result) and relu
      arm_add_f32(matmul_result_, c2_bias_, output_buffer_, out_d_);
      relu(output_buffer_, out_d_);
    }

  private:
    const size_t kernel_size_;
    const size_t in_d_;
    const size_t out_d_;
    float* input_buffer_;
    float* output_buffer_;
    float* c1_kernel_;  // (kernel_size_, in_d, out_d)
    float* c1_bias_;    // (out_d,)
    float* c2_bias_;    // (out_d,)
    arm_matrix_instance_f32 x_mi_;          // (1, in_d)
    arm_matrix_instance_f32 c1_kernel_mi_;  // (in_d, out_d)
    arm_matrix_instance_f32 c2_kernel_mi_;  // (1, out_d, out_d)
    float* matmul_result_; // (out_d,)
    arm_matrix_instance_f32 matmul_result_mi_;
};