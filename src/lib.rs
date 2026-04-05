/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::manual_clamp)]
use num_complex::Complex;
use num_traits::real::Real;
use num_traits::{MulAdd, Num, Zero};
use quefrency::Cepstrum;
use std::fmt::Debug;
use std::ops::{AddAssign, Div, Mul, MulAssign};
use std::sync::Arc;
use zaft::{FftExecutor, R2CFftExecutor, Zaft};

mod cepstrogram;
mod error;
mod frequencies;
mod mel;
mod mla;
mod stft;
mod tempogram;

use crate::cepstrogram::{CepstrogramExecutor, CepstrogramImpl};
use crate::stft::StftExecutorImplReal;
use crate::tempogram::TempogramExecutorImpl;
pub use error::MagspecError;
pub use frequencies::{
    FreqInterpMethod, FreqRemapArgs, remap_freq_log_interp, remap_freq_log_interp_complex,
    remap_freq_log_interp_complex_f64, remap_freq_log_interp_f64,
};
pub use mel::{
    MelFilterbankArgs, MelNorm, MelScale, apply_mel_filterbank, apply_mel_filterbank_complex,
    apply_mel_filterbank_complex_f64, apply_mel_filterbank_f64,
};
pub use tempogram::{TempogramExecutor, TempogramMethod, TempogramOptions};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StftOptions {
    /// FFT size / window length in samples.
    ///
    /// This determines the frequency resolution of the transform.
    pub len: usize,
    /// Number of samples between successive frames.
    ///
    /// Smaller values increase time resolution and overlap.
    pub hop_size: usize,
    /// Window function applied to each frame before the FFT.
    pub window: StftWindow,
    /// If `true`, normalize FFT output by `1 / sqrt(len)`.
    ///
    /// This affects amplitude scaling of the resulting spectrum.
    pub normalize: bool,
    ///  `True` will center DFT cisoids at the window for each shift `u`:
    ///      Sm[u, k] = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
    ///  as opposed to usual STFT:
    ///      S[u, k]  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)
    ///
    ///  Most implementations (including `scipy`, `librosa`) compute *neither*,
    ///  but rather center the window for each slice, thus shifting DFT bases
    ///  relative to n=0 (t=0). These create spectra that, viewed as signals, are
    ///  of high frequency, making inversion and synchrosqueezing very unstable.
    pub modulation: bool,
}

/// Factory for creating Short-Time Fourier Transform (STFT) executors
/// that compute magnitude spectrograms.
pub struct Magspec {}

impl Magspec {
    /// Create a single-precision (f32) STFT magnitude spectrogram executor.
    pub fn make_forward_f32(
        options: StftOptions,
    ) -> Result<Arc<dyn StftExecutor<f32> + Send + Sync>, MagspecError> {
        Ok(Arc::new(StftExecutorImplReal::new(options)?))
    }

    /// Create a double-precision (f64) STFT magnitude spectrogram executor.
    pub fn make_forward_f64(
        options: StftOptions,
    ) -> Result<Arc<dyn StftExecutor<f64> + Send + Sync>, MagspecError> {
        Ok(Arc::new(StftExecutorImplReal::new(options)?))
    }

    /// Creates a single-precision (f32) tempogram executor.
    pub fn make_tempogram_f32(
        options: TempogramOptions,
    ) -> Result<Arc<dyn TempogramExecutor<f32> + Send + Sync>, MagspecError> {
        Ok(Arc::new(TempogramExecutorImpl::new(options)?))
    }

    /// Creates a double-precision (f64) tempogram executor.
    pub fn make_tempogram_f64(
        options: TempogramOptions,
    ) -> Result<Arc<dyn TempogramExecutor<f64> + Send + Sync>, MagspecError> {
        Ok(Arc::new(TempogramExecutorImpl::new(options)?))
    }

    /// Creates a single-precision (f32) cepstrogram executor.
    pub fn make_cepstrogram_f32(
        options: StftOptions,
    ) -> Result<Arc<dyn CepstrogramExecutor<f32> + Send + Sync>, MagspecError> {
        Ok(Arc::new(CepstrogramImpl::new(options)?))
    }

    /// Creates a double-precision (f64) cepstrogram executor.
    pub fn make_cepstrogram_f64(
        options: StftOptions,
    ) -> Result<Arc<dyn CepstrogramExecutor<f64> + Send + Sync>, MagspecError> {
        Ok(Arc::new(CepstrogramImpl::new(options)?))
    }
}

pub(crate) trait StftSample:
    Copy
    + 'static
    + FftFactory
    + WindowFactory
    + Mul<Self, Output = Self>
    + Zero
    + Clone
    + Num
    + Div<Self, Output = Self>
    + MulAssign
    + MulAdd<Self, Output = Self>
    + Real
    + AddAssign
{
}

impl StftSample for f32 {}
impl StftSample for f64 {}

pub(crate) trait FftFactory {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, MagspecError>;
    fn make_c2c(len: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, MagspecError>;
    fn make_cepstrum(len: usize, normalize: bool) -> Result<Cepstrum<Self>, MagspecError>;
}

pub(crate) trait WindowFactory: Sized {
    fn make_window(len: usize, stft_window: StftWindow) -> Vec<Self>;
}

impl WindowFactory for f32 {
    fn make_window(len: usize, stft_window: StftWindow) -> Vec<Self> {
        match stft_window {
            StftWindow::Hann => pxwindow::Pxwindow::hann_f32(len),
            StftWindow::Hamming => pxwindow::Pxwindow::hamming_f32(len),
            StftWindow::Blackman => pxwindow::Pxwindow::blackman_f32(len),
            StftWindow::Slepian { nw } => pxwindow::Pxwindow::slepian_f32(len, nw),
            StftWindow::Kaiser { beta } => pxwindow::Pxwindow::kaiser_f32(len, beta as f32),
        }
    }
}

impl WindowFactory for f64 {
    fn make_window(len: usize, stft_window: StftWindow) -> Vec<Self> {
        match stft_window {
            StftWindow::Hann => pxwindow::Pxwindow::hann_f64(len),
            StftWindow::Hamming => pxwindow::Pxwindow::hamming_f64(len),
            StftWindow::Blackman => pxwindow::Pxwindow::blackman_f64(len),
            StftWindow::Slepian { nw } => pxwindow::Pxwindow::slepian_f64(len, nw),
            StftWindow::Kaiser { beta } => pxwindow::Pxwindow::kaiser_f64(len, beta),
        }
    }
}

impl FftFactory for f32 {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, MagspecError> {
        Zaft::make_r2c_fft_f32(len).map_err(|x| MagspecError::FftError(x.to_string()))
    }

    fn make_c2c(len: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, MagspecError> {
        Zaft::make_forward_fft_f32(len).map_err(|x| MagspecError::FftError(x.to_string()))
    }

    fn make_cepstrum(len: usize, normalize: bool) -> Result<Cepstrum<Self>, MagspecError> {
        quefrency::make_cepstrum_f32(len, normalize)
            .map_err(|x| MagspecError::FftError(x.to_string()))
    }
}

impl FftFactory for f64 {
    fn make_r2c(len: usize) -> Result<Arc<dyn R2CFftExecutor<Self> + Send + Sync>, MagspecError> {
        Zaft::make_r2c_fft_f64(len).map_err(|x| MagspecError::FftError(x.to_string()))
    }

    fn make_c2c(len: usize) -> Result<Arc<dyn FftExecutor<Self> + Send + Sync>, MagspecError> {
        Zaft::make_forward_fft_f64(len).map_err(|x| MagspecError::FftError(x.to_string()))
    }

    fn make_cepstrum(len: usize, normalize: bool) -> Result<Cepstrum<Self>, MagspecError> {
        quefrency::make_cepstrum_f64(len, normalize)
            .map_err(|x| MagspecError::FftError(x.to_string()))
    }
}

#[derive(Debug)]
/// Shared storage type
pub enum BufferStoreMut<'a, T> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T> BufferStoreMut<'_, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

/// Executor trait for computing the Short-Time Fourier Transform (STFT).
///
/// Implementations provide forward computation for real-valued input
/// signals, producing complex spectra or magnitude-only representations.
pub trait StftExecutor<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    /// Allocates a complex-valued output frame for a given input length.
    fn new_complex_frame(
        &self,
        input_len: usize,
    ) -> Result<StftFrameMut<'_, Complex<T>>, MagspecError>;
    /// Allocates a real-valued (magnitude) output frame for a given input length.
    fn new_frame(&self, input_len: usize) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the complex STFT of the input signal.
    fn execute(&self, input: &[T]) -> Result<StftFrameMut<'_, Complex<T>>, MagspecError>;
    /// Computes the complex STFT using a preallocated output frame and scratch buffer.
    fn execute_with_scratch(
        &self,
        input: &[T],
        into: &mut StftFrameMut<'_, Complex<T>>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError>;

    /// Computes the magnitude |X| STFT of the input signal.
    fn execute_magnitude(&self, input: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the magnitude STFT using a preallocated output frame and scratch buffer.
    fn execute_magnitude_with_scratch(
        &self,
        input: &[T],
        into: &mut StftFrameMut<'_, T>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError>;
    /// Returns the required scratch buffer size (in complex elements)
    /// for forward STFT execution.
    fn forward_scratch_size(&self) -> usize;
}

/// Immutable STFT result frame.
///
/// The data is stored in row-major form with:
/// - `width`  = number of time frames
/// - `height` = number of frequency bins
pub struct StftFrame<'a, T>
where
    [T]: ToOwned,
{
    /// Flattened 2D data buffer of size `width * height`.
    pub data: std::borrow::Cow<'a, [T]>,
    /// Number of time frames.
    pub width: usize,
    /// Number of frequency bins.
    pub height: usize,
}

/// Mutable STFT result frame used for zero-allocation execution paths.
pub struct StftFrameMut<'a, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Mutable backing storage of size `width * height`.
    pub data: BufferStoreMut<'a, T>,
    /// Number of time frames.
    pub width: usize,
    /// Number of frequency bins.
    pub height: usize,
}

impl<T> StftFrameMut<'_, T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    pub fn as_ref(&self) -> StftFrame<'_, T> {
        StftFrame {
            data: std::borrow::Cow::Borrowed(self.data.borrow()),
            width: self.width,
            height: self.height,
        }
    }
}

/// Window functions supported by the STFT executor.
#[derive(Clone, Default, PartialOrd, PartialEq, Debug, Copy)]
pub enum StftWindow {
    /// Hann window (default).
    #[default]
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Discrete Prolate Spheroidal Sequences (DPSS) or Slepian Window
    Slepian {
        /// Half-bandwidth, default is 4.0
        nw: f64,
    },
    Kaiser {
        beta: f64,
    },
}
