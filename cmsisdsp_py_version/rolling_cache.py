import math
import numpy as np

class RollingCache(object):
    def __init__(self, depth, dilation, kernel_size):

        # hacky check that dilation is a power of k
        # otherwise this whole thing doesn't make sense
        ne_log_k = math.log(dilation, kernel_size)
        if int(ne_log_k) != ne_log_k:
            raise Exception(f"dilation ({dilation}) should be a power of kernel_size ({kernel_size})")

        self.num_entries = dilation * kernel_size
        self.depth = depth
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.reset_cache()

    def reset_cache(self):
        self.cache = np.zeros((self.num_entries, self.depth), dtype=np.float32)
        self.write_head = 0

    def add(self, x):
        x = x.flatten()
        assert x.shape == (self.depth,), x.shape
        self.write_head += 1
        self.write_head %= self.num_entries
        self.cache[self.write_head] = x

    def cached_dilated_values(self):
        lookup = np.empty((self.kernel_size, self.depth), dtype=np.float32)
        lookup_idx = self.kernel_size - 1
        cache_idx = self.write_head
        while lookup_idx >= 0:
            lookup[lookup_idx] = self.cache[cache_idx]
            lookup_idx -= 1
            cache_idx = (cache_idx - self.dilation) % self.num_entries
        return lookup
