use burn::{backend::Autodiff, tensor::backend::Backend};
use finetune::{inference::infer, training::train};

#[allow(dead_code)]
const ARTIFACT_DIR: &str = "/tmp/resnet-finetune";

#[allow(dead_code)]
fn run<B: Backend>(device: B::Device) {
    train::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    infer::<B>(ARTIFACT_DIR, device, 0.5);
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        super::run::<LibTorch>(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::{
        backend::wgpu::{Wgpu, WgpuDevice},
        Wgpu,
    };

    pub fn run() {
        super::run::<Wgpu>(WgpuDevice::default());
    }
}

fn main() {
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
