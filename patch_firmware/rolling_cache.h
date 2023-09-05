#pragma once

class RollingCache {
  public:

    // cache needs to be preallocated for float[num_entries * depth]
    // ( where num_entries = dilation * kernel_size )
    RollingCache(size_t depth, size_t dilation, size_t kernel_size, float* cache) :
      depth_(depth), dilation_(dilation), kernel_size_(kernel_size),
      cache_(cache), write_head_(0), num_entries_(dilation_ * kernel_size_) {
        ResetCache();
      }

    void ResetCache() {
      for (size_t i=0; i<num_entries_ * depth_; i++ ) {
        cache_[i] = 0;
      }
      write_head_ = 0;
    }

    void Add(float* x) {
      // assume x is float[depth]
      write_head_++;
      write_head_ %= num_entries_;
      const size_t offset = write_head_ * depth_;
      float* dest = &(cache_[offset]);
      memcpy(dest, x, sizeof(float)*depth_);
    }

    void Lookup(float* lookup) {
      // assume lookup is float[kernel_size * depth]
      // write from last to first element.
      // note: int for lookup_idx, not size_t, because we're going backwards
      // and don't want weird 0-1 stuff
      int lookup_idx = kernel_size_-1;
      size_t cache_idx = write_head_;
      while (lookup_idx >= 0) {
        float* dest = &(lookup[lookup_idx * depth_]);
        float* src = &(cache_[cache_idx * depth_]);
        memcpy(dest, src, sizeof(float)*depth_);
        lookup_idx--;
        cache_idx -= dilation_;
        cache_idx %= num_entries_;
      }
    }

  private:
    const size_t depth_;
    const size_t dilation_;
    const size_t kernel_size_;
    float* cache_;
    size_t write_head_;
    const size_t num_entries_;
};