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
use crate::stft::{fftshift_inplace, ifftshift};
use crate::{BufferStoreMut, MagspecError, StftFrameMut, StftOptions, StftSample};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use quefrency::Cepstrum;

/// Executor trait for computing the Short-Time Cepstrogram.
///
/// The cepstrogram is the time-varying extension of the cepstrum:
/// for each analysis frame it computes `IFFT(log(|STFT(x)|))`, producing
/// a 2D representation whose Y-axis is **quefrency** (the cepstral analogue
/// of frequency, measured in samples or seconds) and whose X-axis is time.
pub trait CepstrogramExecutor<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    /// Allocates a real-valued output frame sized for the given input length.
    fn new_frame(&self, input_len: usize) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the cepstrogram of the input signal.
    fn execute(&self, input: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError>;
    /// Computes the cepstrogram using a preallocated output frame and scratch buffer.
    fn execute_with_scratch(
        &self,
        input: &[T],
        into: &mut StftFrameMut<'_, T>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError>;
    /// Returns the minimum scratch buffer size in complex elements required
    /// by [`execute_with_scratch`].
    fn forward_scratch_size(&self) -> usize;
    /// Returns the number of quefrency bins in the output frame.
    fn output_height(&self) -> usize;
    /// Returns the number of time frames in the output for a given input length.
    fn output_width(&self, input_length: usize) -> usize;
}

pub(crate) struct CepstrogramImpl<T> {
    quefrency: Cepstrum<T>,
    /// FFT / window size (must match engine size)
    fft_size: usize,
    /// Number of samples between successive frames (hop)
    hop_size: usize,
    /// If true, normalize each frame by 1/fft_size on inverse
    window: Vec<T>,
    fft_scratch_length: usize,
    modulation: bool,
}

impl<T: StftSample> CepstrogramImpl<T> {
    pub(crate) fn new(options: StftOptions) -> Result<Self, MagspecError> {
        let cepstrum = T::make_cepstrum(options.len, options.normalize)?;
        let mut window = T::make_window(options.len, options.window);
        if options.modulation {
            window = ifftshift(&window);
        }
        let fft_scratch_length = cepstrum.scratch_size();
        Ok(CepstrogramImpl {
            quefrency: cepstrum,
            fft_size: options.len,
            hop_size: options.hop_size.max(1),
            window,
            fft_scratch_length,
            modulation: options.modulation,
        })
    }
}

impl<T: StftSample> CepstrogramExecutor<T> for CepstrogramImpl<T>
where
    f64: AsPrimitive<T>,
    [T]: ToOwned<Owned = Vec<T>>,
    [Complex<T>]: ToOwned<Owned = Vec<Complex<T>>>,
{
    fn new_frame(&self, input_len: usize) -> Result<StftFrameMut<'_, T>, MagspecError> {
        let fft_size = self.fft_size / 2 + 1;
        let width = (input_len - self.fft_size) / self.hop_size + 1;
        let output = try_vec![T::zero(); width * fft_size];
        Ok(StftFrameMut {
            data: BufferStoreMut::Owned(output),
            width,
            height: fft_size,
        })
    }

    fn execute(&self, input: &[T]) -> Result<StftFrameMut<'_, T>, MagspecError> {
        let mut scratch = try_vec![Complex::<T>::zero(); self.forward_scratch_size()];
        let mut frame = self.new_frame(input.len())?;
        self.execute_with_scratch(input, &mut frame, &mut scratch)?;
        Ok(frame)
    }

    fn execute_with_scratch(
        &self,
        r_input: &[T],
        into: &mut StftFrameMut<'_, T>,
        scratch: &mut [Complex<T>],
    ) -> Result<(), MagspecError> {
        let fft_size = self.fft_size;
        let hop_size = self.hop_size;
        let complex_length = fft_size / 2 + 1;
        let fft_scratch_size = self.fft_scratch_length;
        if scratch.len() < self.forward_scratch_size() {
            return Err(MagspecError::InvalidScratchSize(
                scratch.len(),
                self.forward_scratch_size(),
            ));
        }
        let (cut_scratch, _) = scratch.split_at_mut(self.forward_scratch_size());
        let (fft_scratch, rem_scratch0) = cut_scratch.split_at_mut(fft_scratch_size);
        let (_, aligned_mut, _) = unsafe { rem_scratch0.align_to_mut::<T>() };
        let real_working_scratch = &mut aligned_mut[..fft_size];
        let mut input: std::borrow::Cow<'_, [T]> = std::borrow::Cow::Borrowed(r_input);
        if input.len() < self.fft_size {
            let mut new_input = try_vec![T::zero(); fft_size];
            new_input[..r_input.len()].copy_from_slice(input.as_ref());
            input = std::borrow::Cow::Owned(new_input);
        }
        let width = (input.len() - self.fft_size) / self.hop_size + 1;
        if width != into.width {
            return Err(MagspecError::InvalidFrame(
                format_args!(
                    "Invalid frame width, expected {} but it was {}",
                    width, into.width
                )
                .to_string(),
            ));
        }
        if complex_length != into.height {
            return Err(MagspecError::InvalidFrame(
                format_args!(
                    "Invalid frame height, expected {} but it was {}",
                    complex_length, into.height
                )
                .to_string(),
            ));
        }
        if into.data.borrow().len() != width * complex_length {
            return Err(MagspecError::InvalidFrame(
                format_args!(
                    "Invalid frame size, expected {} but it was {}",
                    width * complex_length,
                    into.data.borrow().len()
                )
                .to_string(),
            ));
        }
        for frame in 0..width {
            let start = frame * hop_size;

            if self.modulation {
                let input = if start + fft_size <= input.len() {
                    &input[start..start + fft_size]
                } else {
                    &input[start..]
                };
                real_working_scratch[..input.len()].copy_from_slice(input);
                if input.len() != fft_size {
                    real_working_scratch[input.len()..].fill(T::zero());
                }

                fftshift_inplace(real_working_scratch);

                for (dst, &w) in real_working_scratch.iter_mut().zip(self.window.iter()) {
                    *dst *= w;
                }
            } else {
                let input = if start + fft_size <= input.len() {
                    &input[start..start + fft_size]
                } else {
                    &input[start..]
                };
                for ((dst, &src), &w) in real_working_scratch
                    .iter_mut()
                    .zip(input.iter())
                    .zip(self.window.iter())
                {
                    *dst = src * w;
                }
                if input.len() != fft_size {
                    real_working_scratch[input.len()..].fill(T::zero());
                }
            }

            self.quefrency
                .execute_with_scratch(real_working_scratch, fft_scratch)
                .map_err(|x| MagspecError::FftError(x.to_string()))?;

            let output = into.data.borrow_mut();
            for (i, &input) in real_working_scratch[..complex_length].iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(width * i + frame) = input;
                }
            }
        }
        Ok(())
    }
    #[inline]
    fn forward_scratch_size(&self) -> usize {
        let real_fft_scratch = self.fft_size.div_ceil(2);
        real_fft_scratch + self.fft_scratch_length
    }

    fn output_height(&self) -> usize {
        self.fft_size / 2 + 1
    }

    fn output_width(&self, input_length: usize) -> usize {
        (input_length - self.fft_size) / self.hop_size + 1
    }
}
