#pragma once

class Classifier {
  public:
    Classifier(size_t in_d,
               size_t out_d,
               float* weights,     // (in_d_, out_d_)
               float* biases) :    // (out_d_,)
                in_d_(in_d), out_d_(out_d),
                biases_(biases) {

      // allocate buffer for input
      input_buffer_ = new float[in_d];

      // this matrix instances never changes, so build once
      arm_mat_init_f32(&feature_mi_, 1, in_d_, input_buffer_);
      arm_mat_init_f32(&weights_mi_, in_d, out_d, weights);
    }

  float* GetInputBuffer() { return input_buffer_; }
  const size_t GetInputBufferSize() { return in_d_; }
  const size_t GetOutputBufferSize() { return out_d_; }

  void Apply(float* result) {   // (out_d_,)

      // mat mul features and weights
      arm_matrix_instance_f32 result_mi;
      arm_mat_init_f32(&result_mi, 1, out_d_, result);
      arm_mat_mult_f32(&feature_mi_, &weights_mi_, &result_mi);

      // apply bias
      arm_add_f32(result, biases_, result, out_d_);
  }

  private:
    const size_t in_d_;
    const size_t out_d_;
    float* biases_;
    arm_matrix_instance_f32 feature_mi_;
    arm_matrix_instance_f32 weights_mi_;
    float* input_buffer_;
};