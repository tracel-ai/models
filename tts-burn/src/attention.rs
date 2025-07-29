use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        linear::{Linear, LinearConfig},
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct LocationAwareAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    location_conv: Conv1d<B>,
    location_proj: Linear<B>,
    score_mask_value: f32,
}

impl<B: Backend> LocationAwareAttention<B> {
    pub fn new(
        query_dim: usize,
        key_dim: usize,
        value_dim: usize,
        attention_dim: usize,
        location_kernel_size: usize,
        score_mask_value: f32,
    ) -> Self {
        Self {
            query: LinearConfig::new(query_dim, attention_dim).init(),
            key: LinearConfig::new(key_dim, attention_dim).init(),
            value: LinearConfig::new(value_dim, attention_dim).init(),
            location_conv: Conv1dConfig::new(2, attention_dim, location_kernel_size)
                .with_padding(location_kernel_size / 2)
                .init(),
            location_proj: LinearConfig::new(attention_dim, attention_dim).init(),
            score_mask_value,
        }
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        attention_weights_cum: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let batch_size = query.shape()[0];
        let query_len = query.shape()[1];
        let key_len = key.shape()[1];
        
        // Project inputs
        let query_proj = self.query.forward(query);
        let key_proj = self.key.forward(key);
        let value_proj = self.value.forward(value);
        
        // Compute attention scores
        let scores = query_proj.matmul(key_proj.transpose());
        
        // Add location-aware attention
        if let Some(weights_cum) = attention_weights_cum {
            let location_features = self.location_conv.forward(weights_cum);
            let location_energy = self.location_proj.forward(location_features);
            let location_scores = location_energy.matmul(key_proj.transpose());
            scores = scores + location_scores;
        }
        
        // Apply mask if provided
        if let Some(mask) = mask {
            scores = scores.masked_fill(mask, self.score_mask_value);
        }
        
        // Compute attention weights
        let attention_weights = scores.softmax(-1);
        
        // Apply attention weights to values
        let context = attention_weights.matmul(value_proj);
        
        (context, attention_weights)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    num_heads: usize,
    attention_dim: usize,
    head_dim: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    out_proj: Linear<B>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(
        num_heads: usize,
        attention_dim: usize,
        query_dim: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        let head_dim = attention_dim / num_heads;
        
        Self {
            num_heads,
            attention_dim,
            head_dim,
            query: LinearConfig::new(query_dim, attention_dim).init(),
            key: LinearConfig::new(key_dim, attention_dim).init(),
            value: LinearConfig::new(value_dim, attention_dim).init(),
            out_proj: LinearConfig::new(attention_dim, attention_dim).init(),
        }
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let batch_size = query.shape()[0];
        let query_len = query.shape()[1];
        let key_len = key.shape()[1];
        
        // Project inputs
        let query_proj = self.query.forward(query);
        let key_proj = self.key.forward(key);
        let value_proj = self.value.forward(value);
        
        // Reshape for multi-head attention
        let query_heads = query_proj
            .reshape([batch_size, query_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let key_heads = key_proj
            .reshape([batch_size, key_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let value_heads = value_proj
            .reshape([batch_size, key_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        
        // Compute attention scores
        let scores = query_heads.matmul(key_heads.transpose(-1, -2));
        let scores = scores / (self.head_dim as f32).sqrt();
        
        // Apply mask if provided
        if let Some(mask) = mask {
            scores = scores.masked_fill(mask, f32::NEG_INFINITY);
        }
        
        // Compute attention weights
        let attention_weights = scores.softmax(-1);
        
        // Apply attention weights to values
        let context = attention_weights.matmul(value_heads);
        
        // Reshape and project output
        let context = context
            .transpose(1, 2)
            .reshape([batch_size, query_len, self.attention_dim]);
        
        self.out_proj.forward(context)
    }
} 