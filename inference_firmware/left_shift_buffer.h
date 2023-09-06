#pragma once

class LeftShiftBuffer {
  public:
    LeftShiftBuffer(size_t width, size_t depth) :
      width_(width), depth_(depth), buffer_(0) {}

  void Add(float* new_entry) {
    // warning! calling without set will crash.
    // shift all entries down
    for (size_t i=0; i < (width_-1)*depth_; i++) {
      buffer_[i] = buffer_[i + depth_];
    }
    // write in new entry
    int offset = (width_-1) * depth_;
    for (size_t i=0; i<depth_; i++) {
      buffer_[offset+i] = new_entry[i];
    }
  }

  void SetOutputBuffer(float* buffer) { buffer_ = buffer; }
  const size_t GetOutputBufferSize() { return width_ * depth_; }

  private:
    const size_t width_;
    const size_t depth_;
    float* buffer_;
};