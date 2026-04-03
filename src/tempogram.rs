/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::error::try_vec;
use crate::mla::fmla;
use crate::stft::StftExecutorImplReal;
use crate::{
    BufferStoreMut, MagspecError, StftExecutor, StftFrameMut, StftOptions, StftSample, StftWindow,
};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;
use zaft::FftExecutor;

pub struct TempogramOptions {
    /// STFT options for computing the onset envelope
    pub stft: StftOptions,
    /// Window size for the autocorrelation/FFT analysis (in onset frames)
    pub tempo_window_size: usize,
    /// Hop size for the tempogram (in onset frames)
    pub tempo_hop_size: usize,
    /// Whether to use FFT-based (Fourier tempogram) or autocorrelation (cyclic tempogram)
    pub method: TempogramMethod,
    /// Whether to normalize the output
    pub normalize: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempogramMethod {
    /// Autocorrelation tempogram — output Y-axis is lag (in onset frames)
    Autocorrelation,
    /// Fourier tempogram — output Y-axis is frequency (cycles per onset frame)
    Fourier,
}

/// Executor trait for computing the Tempogram.
///
/// The tempogram is the time-varying representation of rhythmic periodicity.
/// It is the tempo-domain analogue of the spectrogram: where a spectrogram
/// reveals how frequency content evolves over time, the tempogram reveals
/// how **tempo** (or more generally, any periodicity) evolves over time.
pub trait TempogramExecutor<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    /// Allocates a real-valued output frame sized for the given onset envelope length.
    fn new_frame(&self, onset_len: usize) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the tempogram of a raw input signal.
    fn execute(&self, input: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the tempogram directly from a pre-computed onset envelope.
    fn execute_from_onset(&self, onset_env: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Returns the minimum scratch buffer size in complex elements required
    /// for internal computation.
    fn scratch_size(&self) -> usize;
    /// Returns the number of rows in the output frame.
    fn output_height(&self) -> usize;
    /// Returns the number of time frames in the output for a given onset envelope length.
    fn output_width(&self, onset_len: usize) -> usize;
}

pub(crate) struct TempogramExecutorImpl<T> {
    stft: StftExecutorImplReal<T>,
    tempo_window: Vec<T>,
    tempo_window_size: usize,
    tempo_hop_size: usize,
    method: TempogramMethod,
    normalize: bool,
    fft_c2c: Option<Arc<dyn FftExecutor<T> + Send + Sync>>,
    fft_scratch_length: usize,
}

impl<T: StftSample> TempogramExecutor<T> for TempogramExecutorImpl<T>
where
    f64: AsPrimitive<T>,
    [T]: ToOwned<Owned = Vec<T>>,
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    fn new_frame(&self, onset_len: usize) -> Result<StftFrameMut<'_, T>, MagspecError> {
        let height = self.output_height();
        let width = self.output_width(onset_len);
        let output = try_vec![T::zero(); width * height];
        Ok(StftFrameMut {
            data: BufferStoreMut::Owned(output),
            width,
            height,
        })
    }

    fn execute(&self, input: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>
    where
        f64: AsPrimitive<T>,
    {
        let mag = self.stft.execute_magnitude(input)?;
        let env = self.onset_envelope(&mag)?;

        let mut frame = self.new_frame(env.len())?;
        let mut scratch = try_vec![Complex::<T>::zero(); self.scratch_size()];

        self.execute_with_scratch_inner(&env, &mut frame, &mut scratch)?;
        Ok(frame)
    }

    fn execute_from_onset(&self, onset_env: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>
    where
        f64: AsPrimitive<T>,
    {
        let mut frame = self.new_frame(onset_env.len())?;
        let mut scratch = try_vec![Complex::<T>::zero(); self.scratch_size()];
        self.execute_with_scratch_inner(onset_env, &mut frame, &mut scratch)?;
        Ok(frame)
    }

    fn scratch_size(&self) -> usize {
        match self.method {
            TempogramMethod::Fourier => {
                self.fft_scratch_length + self.tempo_window_size + self.tempo_window_size / 2 + 1
            }
            TempogramMethod::Autocorrelation => self.tempo_window_size,
        }
    }

    fn output_height(&self) -> usize {
        match self.method {
            TempogramMethod::Autocorrelation => self.tempo_window_size,
            TempogramMethod::Fourier => self.tempo_window_size / 2 + 1,
        }
    }

    fn output_width(&self, onset_len: usize) -> usize {
        if onset_len < self.tempo_window_size {
            1
        } else {
            (onset_len - self.tempo_window_size) / self.tempo_hop_size + 1
        }
    }
}

impl<T: StftSample> TempogramExecutorImpl<T>
where
    f64: AsPrimitive<T>,
    [T]: ToOwned<Owned = Vec<T>>,
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    pub(crate) fn new(options: TempogramOptions) -> Result<Self, MagspecError> {
        let stft = StftExecutorImplReal::new(options.stft)?;
        let tempo_window_size = options.tempo_window_size.max(1);
        let tempo_hop_size = options.tempo_hop_size.max(1);

        let tempo_window = T::make_window(tempo_window_size, StftWindow::Hann);

        let (fft_c2c, fft_scratch_length) = match options.method {
            TempogramMethod::Fourier => {
                let fft = T::make_c2c(tempo_window_size)?;
                let scratch_len = fft.scratch_length();
                (Some(fft), scratch_len)
            }
            TempogramMethod::Autocorrelation => (None, 0),
        };

        Ok(TempogramExecutorImpl {
            stft,
            tempo_window,
            tempo_window_size,
            tempo_hop_size,
            method: options.method,
            normalize: options.normalize,
            fft_c2c,
            fft_scratch_length,
        })
    }

    /// Half-wave rectified spectral flux from a magnitude STFT frame.
    fn onset_envelope(&self, mag: &StftFrameMut<'_, T>) -> Result<Vec<T>, MagspecError> {
        let width = mag.width;
        let height = mag.height;
        let data = mag.data.borrow();
        let mut env = try_vec![T::zero(); width];
        for (t, dst) in (1..width).zip(env[1..width].iter_mut()) {
            let mut flux = T::zero();
            for f in 0..height {
                let prev = data[width * f + (t - 1)];
                let curr = data[width * f + t];
                let diff = curr - prev;
                if diff > T::zero() {
                    flux += diff;
                }
            }
            *dst = flux;
        }
        Ok(env)
    }

    /// Fill one tempogram column using autocorrelation.
    /// `col` has length `tempo_window_size` (= number of lags).
    fn fill_autocorrelation_col(&self, window_slice: &[T], col: &mut [T], scratch: &mut [T]) {
        let w = window_slice.len();
        let ws = scratch.len().min(w);

        // Apply Hann window into scratch
        for i in 0..ws {
            scratch[i] = window_slice[i] * self.tempo_window[i];
        }
        scratch[ws..].fill(T::zero());

        let n = self.tempo_window_size;
        let biased_scale: T = (1.0_f64 / n as f64).as_();

        let energy: T = scratch.iter().fold(T::zero(), |a, &x| a + x * x) * biased_scale;

        let norm = if self.normalize {
            if energy > T::zero() {
                T::one() / energy
            } else {
                T::one()
            }
        } else {
            T::one()
        };

        for (lag, out) in col.iter_mut().enumerate() {
            if lag >= n {
                *out = T::zero();
                continue;
            }
            let acf: T = scratch[..n - lag]
                .iter()
                .zip(scratch[lag..].iter())
                .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
                * biased_scale;
            *out = acf * norm;
        }
    }

    fn fill_fourier_col(
        &self,
        window_slice: &[T],
        col: &mut [T],
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError> {
        let w = self.tempo_window_size;
        let (complex_input, fft_scratch) = scratch.split_at_mut(w);

        let n = window_slice.len();
        let rcp = (1f64 / n as f64).as_();
        let mean: T = if n > 0 {
            window_slice.iter().fold(T::zero(), |a, &x| a + x) * rcp
        } else {
            T::zero()
        };

        for (i, dst) in complex_input.iter_mut().enumerate() {
            let sample = if i < window_slice.len() {
                (window_slice[i] - mean) * self.tempo_window[i]
            } else {
                T::zero()
            };
            *dst = Complex::new(sample, T::zero());
        }

        self.fft_c2c
            .as_ref()
            .expect("Fourier method requires fft_c2c")
            .execute_with_scratch(complex_input, fft_scratch)
            .map_err(|x| MagspecError::FftError(x.to_string()))?;

        let one_sided = w / 2 + 1;
        let norm_factor: T = if self.normalize {
            (1.0_f64 / w as f64).as_()
        } else {
            T::one()
        };
        for (i, out) in col[..one_sided].iter_mut().enumerate() {
            let c = complex_input[i];
            *out = fmla(c.re, c.re, c.im * c.im).sqrt() * norm_factor;
        }
        Ok(())
    }

    fn execute_with_scratch_inner(
        &self,
        env: &[T],
        into: &mut StftFrameMut<'_, T>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError>
    where
        f64: AsPrimitive<T>,
    {
        let width = into.width;

        for frame_idx in 0..width {
            let start = frame_idx * self.tempo_hop_size;
            let end = (start + self.tempo_window_size).min(env.len());
            let slice = &env[start..end];

            let output = into.data.borrow_mut();

            match self.method {
                TempogramMethod::Autocorrelation => {
                    let (_, casted_scratch, _) = unsafe { scratch.align_to_mut::<T>() };
                    let t_scratch = &mut casted_scratch[..self.tempo_window_size * 2];
                    let (work, col) = t_scratch.split_at_mut(self.tempo_window_size);
                    self.fill_autocorrelation_col(slice, col, work);
                    for (row, &val) in col.iter().enumerate() {
                        unsafe {
                            *output.get_unchecked_mut(width * row + frame_idx) = val;
                        }
                    }
                }
                TempogramMethod::Fourier => {
                    let one_sided = self.tempo_window_size / 2 + 1;
                    let (scratch, rem_scratch) =
                        scratch.split_at_mut(self.fft_scratch_length + self.tempo_window_size);
                    let (_, q_casted_col, _) = unsafe { rem_scratch.align_to_mut::<T>() };
                    let col = &mut q_casted_col[..one_sided];

                    self.fill_fourier_col(slice, col, scratch)?;

                    let output = into.data.borrow_mut();
                    for (row, &val) in col.iter().enumerate() {
                        unsafe {
                            *output.get_unchecked_mut(width * row + frame_idx) = val;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
