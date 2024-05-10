use burn::tensor::{backend::Backend, ElementConversion, Int, Tensor};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};

pub enum Sampler {
    TopP(TopP),
    Argmax,
}

impl Sampler {
    pub fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        match self {
            Self::TopP(s) => s.sample(logits),
            Self::Argmax => logits.argmax(1),
        }
    }
}

pub trait Sampling {
    fn sample<B: Backend>(&mut self, logits: Tensor<B, 2>) -> Tensor<B, 2, Int>;
}

/// Top-p sampling (nucleus sampling) selects the smallest set of tokens whose cumulative
/// probability mass exceed the threshold p.
pub struct TopP {
    /// Probability threshold for sampling.
    p: f64,
    /// RNG.
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f64, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { p, rng }
    }
}

impl Sampling for TopP {
    fn sample<B: Backend>(&mut self, probs: Tensor<B, 2>) -> Tensor<B, 2, Int> {
        assert_eq!(
            probs.dims()[0],
            1,
            "Naive top-p sampling only supports single-batch tensors"
        );
        let device = probs.device();
        let probs_sort = probs.sort_descending(1);

        // TODO: cumsum
        // let (probs_sort, probs_idx) = probs.sort_descending_with_indices(2);
        // probs_sum = probs_sort.cumsum_dim(2);
        // Clamp smaller probabilities to zero
        // let mask = (probs_sum - probs_sort).greater_elem(self.p);
        // probs_sort.mask_fill(mask, 0.0);
        // let probs_sort = probs_sort / probs_sort.sum_dim(1);

        // TODO: Distribution::Multinomial (aka https://docs.rs/rand/latest/rand/distributions/struct.WeightedIndex.html)
        // let next_token = multinomial

        let mut probs_sort = probs_sort
            .to_data()
            .value
            .iter()
            .map(|e| e.elem::<f64>())
            .collect::<Vec<_>>();

        let mut cumsum = 0.;
        probs_sort.iter_mut().for_each(|x| {
            if cumsum >= self.p {
                *x = 0.0;
            } else {
                cumsum += *x;
            }
        });
        // .scan(0.0, |acc, (idx, x)| {
        //     *acc = *acc + x;
        //     // Clamp smaller probabilities to zero.
        //     if *acc >= self.p {
        //         probs_sort[idx] = 0.0.elem();
        //     }
        //     Some(*acc)
        // })
        // .collect::<Vec<_>>();

        let next_token = WeightedIndex::new(probs_sort)
            .unwrap()
            .sample(&mut self.rng) as i32;

        Tensor::<B, 2, Int>::from_ints([[next_token]], &device)
    }
}
