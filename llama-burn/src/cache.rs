use burn::tensor::{backend::Backend, Tensor};

/// All Llama-3 models support sequence length up to 8192 tokens.
pub(crate) const MAX_SEQ_LEN: usize = 8192;

// /// All Llama-2 models support sequence length up to 4096 tokens.
// pub(crate) const MAX_SEQ_LEN_V2: usize = 4096;

// Adapted from `burn::nn::cache`
enum CacheState<T> {
    Value(T),
    Empty,
}

/// A cache for a tensor.
struct TensorCache<B: Backend, const D: usize> {
    state: CacheState<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    /// Creates a new empty cache.
    ///
    /// # Returns
    ///
    /// The empty cache.
    pub fn empty() -> Self {
        Self {
            state: CacheState::Empty,
        }
    }
}

pub(crate) struct AutoregressiveCache<B: Backend> {
    /// Tensor cache with shape `[max_batch_size, num_heads, seq_len, head_dim]`
    cache: TensorCache<B, 4>,
    pub(crate) max_seq_len: usize,
}

impl<B: Backend> AutoregressiveCache<B> {
    /// Creates a new empty cache.
    pub fn new(max_seq_len: usize) -> Self {
        assert!(
            max_seq_len <= MAX_SEQ_LEN,
            "Maximum sequence length must not exceed {MAX_SEQ_LEN}"
        );

        Self {
            cache: TensorCache::empty(),
            max_seq_len,
        }
    }

    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut tensor_old = CacheState::Empty;
        core::mem::swap(&mut self.cache.state, &mut tensor_old);

        let tensor_new = match tensor_old {
            CacheState::Value(tensor_old) => {
                let mut tensor_new = Tensor::cat(vec![tensor_old, tensor], 2);

                // Limit to context length (aka max sequence length)
                let seq_len = tensor_new.dims()[2];
                if seq_len > self.max_seq_len {
                    tensor_new = tensor_new.narrow(2, seq_len - self.max_seq_len, self.max_seq_len);
                }
                tensor_new
            }
            _ => tensor,
        };

        self.cache.state = CacheState::Value(tensor_new.clone());
        tensor_new
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        match &self.cache.state {
            CacheState::Empty => 0,
            CacheState::Value(t) => t.dims()[2],
        }
    }
}
