#![cfg_attr(not(feature = "std"), no_std)]
mod block;
#[cfg(feature = "train")]
mod training;

pub mod resnet;
pub mod weights;

pub use resnet::*;
#[cfg(feature = "train")]
pub use training::*;
pub use weights::*;

extern crate alloc;
