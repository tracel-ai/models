use burn::tensor::{backend::Backend, Tensor};
use std::f32::consts::PI;

pub struct STFT {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Vec<f32>,
}

impl STFT {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        let window = Self::hann_window(win_length);
        Self {
            n_fft,
            hop_length,
            win_length,
            window,
        }
    }

    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    pub fn forward<B: Backend>(&self, audio: &[f32]) -> Tensor<B, 3> {
        let n_frames = (audio.len() as f32 / self.hop_length as f32).ceil() as usize;
        let mut stft = Vec::with_capacity(n_frames * self.n_fft);
        
        for i in 0..n_frames {
            let start = i * self.hop_length;
            let end = (start + self.win_length).min(audio.len());
            
            // Apply window and zero-pad if necessary
            let mut frame = vec![0.0; self.n_fft];
            for j in 0..(end - start) {
                frame[j] = audio[start + j] * self.window[j];
            }
            
            // Compute FFT
            let fft = self.compute_fft(&frame);
            stft.extend(fft);
        }
        
        Tensor::from_data(stft)
            .reshape([1, n_frames, self.n_fft])
    }

    fn compute_fft(&self, frame: &[f32]) -> Vec<f32> {
        // This is a simplified FFT implementation
        // In practice, you would use a proper FFT library
        let n = frame.len();
        let mut real = frame.to_vec();
        let mut imag = vec![0.0; n];
        
        // Cooley-Tukey FFT
        for i in 0..n {
            let mut j = 0;
            for k in 0..n {
                j = (j << 1) | (k & 1);
                j >>= 1;
            }
            if j > i {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }
        
        // Combine real and imaginary parts
        real.iter()
            .zip(imag.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect()
    }
}

pub struct MelFilterbank {
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    mel_basis: Vec<Vec<f32>>,
}

impl MelFilterbank {
    pub fn new(n_mels: usize, n_fft: usize, sample_rate: u32) -> Self {
        let mel_basis = Self::create_mel_basis(n_mels, n_fft, sample_rate);
        Self {
            n_mels,
            n_fft,
            sample_rate,
            mel_basis,
        }
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_basis(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f32>> {
        let f_min = 0.0;
        let f_max = sample_rate as f32 / 2.0;
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);
        
        let mel_points = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect::<Vec<_>>();
        
        let fft_freqs = (0..n_fft / 2 + 1)
            .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
            .collect::<Vec<_>>();
        
        let mut basis = vec![vec![0.0; n_fft / 2 + 1]; n_mels];
        
        for i in 0..n_mels {
            let f_left = Self::mel_to_hz(mel_points[i]);
            let f_center = Self::mel_to_hz(mel_points[i + 1]);
            let f_right = Self::mel_to_hz(mel_points[i + 2]);
            
            for j in 0..n_fft / 2 + 1 {
                let freq = fft_freqs[j];
                if freq >= f_left && freq <= f_right {
                    if freq <= f_center {
                        basis[i][j] = (freq - f_left) / (f_center - f_left);
                    } else {
                        basis[i][j] = (f_right - freq) / (f_right - f_center);
                    }
                }
            }
        }
        
        basis
    }

    pub fn forward<B: Backend>(&self, stft: Tensor<B, 3>) -> Tensor<B, 3> {
        let shape = stft.shape();
        let batch_size = shape[0];
        let n_frames = shape[1];
        
        let mut mel_spec = Vec::with_capacity(batch_size * n_frames * self.n_mels);
        
        for b in 0..batch_size {
            for t in 0..n_frames {
                for m in 0..self.n_mels {
                    let mut mel_value = 0.0;
                    for k in 0..self.n_fft / 2 + 1 {
                        mel_value += stft[[b, t, k]] * self.mel_basis[m][k];
                    }
                    mel_spec.push(mel_value);
                }
            }
        }
        
        Tensor::from_data(mel_spec)
            .reshape([batch_size, n_frames, self.n_mels])
    }
} 