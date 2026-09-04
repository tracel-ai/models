use burn::tensor::{debug_assert_shape, Device, Tensor};

pub(crate) struct AutoregressiveCache {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: Tensor<4>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: usize,
}

impl AutoregressiveCache {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device,
    ) -> Self {
        Self {
            cache: Tensor::empty([max_batch_size, num_heads, max_seq_len, d_model], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    pub fn forward(&mut self, tensor: Tensor<4>) -> Tensor<4> {
        // The batch may be smaller than the cache was built for, but the head and feature
        // axes have to line up or the slice_assign below writes into the wrong cells.
        let [_, cache_heads, _, cache_d_model] = self.cache.dims();
        debug_assert_shape!(tensor, [_, cache_heads, _, cache_d_model]);

        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor,
        );

        self.cur_seq_len += seq_len;

        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model])
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}
