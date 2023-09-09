#pragma once

class Regression {
  public:
    Regression(size_t in_d,
               size_t out_d,
               float* weights,     // (in_d_, out_d_)
               float* biases) :
                in_d_(in_d), out_d_(out_d),
                biases_(biases) {

      // allocate buffer for input and output
      input_buffer_ = new float[in_d];
      output_buffer_ = new float[out_d];

      // matrix instances never changes, so build once
      arm_mat_init_f32(&feature_mi_, 1, in_d_, input_buffer_);
      arm_mat_init_f32(&weights_mi_, in_d, out_d, weights);
      arm_mat_init_f32(&result_mi_, 1, out_d_, output_buffer_);
    }

  float* GetInputBuffer() { return input_buffer_; }
  const size_t GetInputBufferSize() { return in_d_; }
  float* GetOutputBuffer() { return output_buffer_; }
  const size_t GetOutputBufferSize() { return out_d_; }

  void Run() {
      // mat mul features and weights
      arm_mat_mult_f32(&feature_mi_, &weights_mi_, &result_mi_);
      // add bias
      arm_add_f32(output_buffer_, biases_, output_buffer_, out_d_);
  }

  private:
    const size_t in_d_;
    const size_t out_d_;
    float* biases_;
    arm_matrix_instance_f32 feature_mi_;
    arm_matrix_instance_f32 weights_mi_;
    arm_matrix_instance_f32 result_mi_;
    float* input_buffer_;
    float* output_buffer_;
};