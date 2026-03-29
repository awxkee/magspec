/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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

/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::frequencies::FreqBlend;
use crate::mla::fmla;
use crate::{BufferStoreMut, MagspecError, StftFrame, StftFrameMut};
use num_complex::Complex;
use pxfm::{f_exp10f, f_expf, f_log10f, f_logf};

pub(crate) trait MelAccum: FreqBlend {
    fn mel_accum(acc: &mut Self, weight: f32, value: Self);
}

impl MelAccum for f32 {
    #[inline(always)]
    fn mel_accum(acc: &mut Self, weight: f32, value: Self) {
        *acc = fmla(weight, value, *acc);
    }
}

impl MelAccum for f64 {
    #[inline(always)]
    fn mel_accum(acc: &mut Self, weight: f32, value: Self) {
        *acc = fmla(weight as f64, value, *acc);
    }
}

impl MelAccum for Complex<f32> {
    #[inline(always)]
    fn mel_accum(acc: &mut Self, weight: f32, value: Self) {
        acc.re = fmla(weight, value.re, acc.re);
        acc.im = fmla(weight, value.im, acc.im);
    }
}

impl MelAccum for Complex<f64> {
    #[inline(always)]
    fn mel_accum(acc: &mut Self, weight: f32, value: Self) {
        let w = weight as f64;
        acc.re = fmla(w, value.re, acc.re);
        acc.im = fmla(w, value.im, acc.im);
    }
}

/// Mel scale formula to use when converting between Hz and Mel.
///
/// The two variants differ in how they place the break-point between the
/// linear and logarithmic segments of the scale.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MelScale {
    /// HTK formula: `mel = 2595 * log10(1 + hz / 700)`.
    #[default]
    Htk,
    /// Slaney
    Slaney,
}

impl MelScale {
    #[inline]
    fn hz_to_mel(self, hz: f32) -> f32 {
        match self {
            MelScale::Htk => 2595.0 * f_log10f(1.0 + hz / 700.0),
            MelScale::Slaney => {
                const F_SP: f32 = 200.0 / 3.0;
                const MIN_LOG_HZ: f32 = 1000.0;
                const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - 0.0) / F_SP; // = 15.0
                const LOG_STEP: f32 = 0.0687517774209491174994593064468405; // ln(6.4) / 27

                if hz < MIN_LOG_HZ {
                    hz / F_SP
                } else {
                    MIN_LOG_MEL + f_logf(hz / MIN_LOG_HZ) / LOG_STEP
                }
            }
        }
    }

    #[inline]
    fn mel_to_hz(self, mel: f32) -> f32 {
        match self {
            MelScale::Htk => 700.0 * (f_exp10f(mel / 2595.0) - 1.0),
            MelScale::Slaney => {
                const F_SP: f32 = 200.0 / 3.0;
                const MIN_LOG_HZ: f32 = 1000.0;
                const MIN_LOG_MEL: f32 = 15.0;
                const LOG_STEP: f32 = 0.0687517774209491174994593064468405; // ln(6.4) / 27

                if mel < MIN_LOG_MEL {
                    mel * F_SP
                } else {
                    MIN_LOG_HZ * f_expf((mel - MIN_LOG_MEL) * LOG_STEP)
                }
            }
        }
    }
}

// ── Filter normalisation ──────────────────────────────────────────────────────

/// How to normalise each triangular Mel filter.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MelNorm {
    /// No normalisation; each filter sums its triangle weights as-is.
    /// Energy is proportional to the bandwidth of the filter.
    #[default]
    None,
    /// Divide each filter by its bandwidth in Hz
    Slaney,
}

#[derive(Debug, Clone, Copy)]
pub struct MelFilterbankArgs {
    /// Sample rate of the original audio signal in Hz.
    pub sample_rate: f32,
    /// FFT size used to produce the STFT magnitude frame.
    /// Determines the Hz-per-bin spacing as `sample_rate / fft_size`.
    pub fft_size: usize,
    /// Number of Mel filter bands in the output.
    /// Typical values: 40, 80, 128.
    pub num_mel_bins: usize,
    /// Lowest frequency covered by the filterbank in Hz.
    /// Use `0.0` to start from DC (bin 0).
    pub f_min: f32,
    /// Highest frequency covered by the filterbank in Hz.
    /// `None` defaults to the Nyquist frequency (`sample_rate / 2`).
    pub f_max: Option<f32>,
    /// Filter normalization strategy.
    pub norm: MelNorm,
    /// Mel scale formula.
    pub scale: MelScale,
}

impl MelFilterbankArgs {
    /// Apply the filterbank to a **magnitude** (real-valued) STFT frame.
    pub fn apply<'a>(
        &self,
        frame: &StftFrame<'a, f32>,
    ) -> Result<StftFrameMut<'static, f32>, MagspecError> {
        apply_mel_filterbank(frame, self)
    }

    /// Build and return the raw filterbank matrix.
    pub fn build_filterbank(&self) -> Result<Vec<f32>, MagspecError> {
        let f_max = self.f_max.unwrap_or(self.sample_rate / 2.0);
        let num_freq_bins = self.fft_size / 2 + 1;
        build_filterbank_matrix(
            self.sample_rate,
            self.fft_size,
            num_freq_bins,
            self.num_mel_bins,
            self.f_min,
            f_max,
            self.norm,
            self.scale,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn build_filterbank_matrix(
    sample_rate: f32,
    fft_size: usize,
    num_freq_bins: usize,
    num_mel_bins: usize,
    f_min: f32,
    f_max: f32,
    norm: MelNorm,
    scale: MelScale,
) -> Result<Vec<f32>, MagspecError> {
    if sample_rate <= 0.0 || !sample_rate.is_finite() {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "sample_rate must be finite and > 0, got {sample_rate}"
        )));
    }
    if fft_size == 0 {
        return Err(MagspecError::InvalidRemapArgs(
            "fft_size must be > 0".into(),
        ));
    }
    // fft_size must be a power of two — virtually all FFT engines require this.
    if !fft_size.is_power_of_two() {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "fft_size must be a power of two, got {fft_size}"
        )));
    }
    if num_freq_bins == 0 {
        return Err(MagspecError::InvalidRemapArgs(
            "num_freq_bins must be > 0".into(),
        ));
    }
    // Caller should pass fft_size/2+1; catch obvious mismatches.
    let expected_freq_bins = fft_size / 2 + 1;
    if num_freq_bins != expected_freq_bins {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "num_freq_bins ({num_freq_bins}) does not match fft_size/2+1 ({expected_freq_bins})"
        )));
    }
    if num_mel_bins == 0 {
        return Err(MagspecError::InvalidRemapArgs(
            "num_mel_bins must be > 0".into(),
        ));
    }
    if !f_min.is_finite() || f_min < 0.0 {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_min must be finite and >= 0, got {f_min}"
        )));
    }
    if !f_max.is_finite() || f_max <= 0.0 {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max must be finite and > 0, got {f_max}"
        )));
    }
    if f_max <= f_min {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max ({f_max}) must be strictly greater than f_min ({f_min})"
        )));
    }
    let nyquist = sample_rate / 2.0;
    if f_max > nyquist + 1e-3 {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max ({f_max}) exceeds Nyquist frequency ({nyquist})"
        )));
    }
    // f_min must be strictly below f_max (already checked) and also below Nyquist.
    if f_min >= nyquist {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_min ({f_min}) must be below Nyquist ({nyquist})"
        )));
    }

    // num_mel_bins + 2 equally-spaced Mel points covering [f_min, f_max],
    let mel_min = scale.hz_to_mel(f_min);
    let mel_max = scale.hz_to_mel(f_max);

    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| {
            let t = i as f32 / (num_mel_bins + 1) as f32;
            scale.mel_to_hz(mel_min + t * (mel_max - mel_min))
        })
        .collect();

    let bin_hz = sample_rate / fft_size as f32;
    let f_bins: Vec<f32> = mel_points
        .iter()
        .map(|&hz| (hz / bin_hz).clamp(0.0, (num_freq_bins - 1) as f32))
        .collect();

    // Build the matrix: one triangular filter per Mel band.
    let mut matrix = try_vec![0f32; num_mel_bins * num_freq_bins];

    for m in 0..num_mel_bins {
        let f_left = f_bins[m];
        let f_centre = f_bins[m + 1];
        let f_right = f_bins[m + 2];

        let row = &mut matrix[m * num_freq_bins..(m + 1) * num_freq_bins];

        // Rising slope
        let lo = f_left.floor() as usize;
        let hi = f_centre.floor() as usize;
        for (k, dst) in row
            .iter_mut()
            .enumerate()
            .take(hi.min(num_freq_bins - 1) + 1)
            .skip(lo)
        {
            let kf = k as f32;
            let denom = f_centre - f_left;
            if denom > 1e-8 {
                *dst = ((kf - f_left) / denom).max(0.0).min(1.0);
            } else {
                *dst = 1.0; // degenerate single-bin filter
            }
        }

        // Falling slope
        let lo2 = f_centre.floor() as usize + 1; // start one past centre
        let hi2 = f_right.ceil() as usize;
        for (k, dst) in row
            .iter_mut()
            .enumerate()
            .take(hi2.min(num_freq_bins - 1) + 1)
            .skip(lo2)
        {
            let kf = k as f32;
            let denom = f_right - f_centre;
            if denom > 1e-8 {
                *dst = ((f_right - kf) / denom).max(0.0).min(1.0);
            }
        }

        // Normalization
        if norm == MelNorm::Slaney {
            let bandwidth_hz = mel_points[m + 2] - mel_points[m];
            if bandwidth_hz > 1e-8 {
                let inv_bw = 2.0 / bandwidth_hz;
                for v in row.iter_mut() {
                    *v *= inv_bw;
                }
            }
        }
    }

    Ok(matrix)
}

/// Apply a Mel filterbank to a magnitude STFT frame.
pub fn apply_mel_filterbank_complex(
    frame: &StftFrame<'_, Complex<f32>>,
    args: &MelFilterbankArgs,
) -> Result<StftFrameMut<'static, Complex<f32>>, MagspecError> {
    apply_mel_filterbank_impl(frame, args)
}

/// Apply a Mel filterbank to a magnitude STFT frame.
pub fn apply_mel_filterbank_complex_f64(
    frame: &StftFrame<'_, Complex<f64>>,
    args: &MelFilterbankArgs,
) -> Result<StftFrameMut<'static, Complex<f64>>, MagspecError> {
    apply_mel_filterbank_impl(frame, args)
}

/// Apply a Mel filterbank to a magnitude STFT frame.
pub fn apply_mel_filterbank(
    frame: &StftFrame<'_, f32>,
    args: &MelFilterbankArgs,
) -> Result<StftFrameMut<'static, f32>, MagspecError> {
    apply_mel_filterbank_impl(frame, args)
}

/// Apply a Mel filterbank to a magnitude STFT frame.
pub fn apply_mel_filterbank_f64(
    frame: &StftFrame<'_, f64>,
    args: &MelFilterbankArgs,
) -> Result<StftFrameMut<'static, f64>, MagspecError> {
    apply_mel_filterbank_impl(frame, args)
}

/// Apply a Mel filterbank to a magnitude STFT frame.
fn apply_mel_filterbank_impl<T: MelAccum>(
    frame: &StftFrame<'_, T>,
    args: &MelFilterbankArgs,
) -> Result<StftFrameMut<'static, T>, MagspecError> {
    let num_frames = frame.width;
    let num_freq_bins = frame.height;

    if num_frames == 0 || num_freq_bins == 0 {
        return Err(MagspecError::InvalidFrame(format!(
            "frame dimensions are zero (width={}, height={})",
            num_frames, num_freq_bins
        )));
    }
    if frame.data.len() != num_frames * num_freq_bins {
        return Err(MagspecError::InvalidFrame(format!(
            "data length {} does not match width * height = {}",
            frame.data.len(),
            num_frames * num_freq_bins
        )));
    }
    if args.fft_size == 0 {
        return Err(MagspecError::InvalidRemapArgs(
            "fft_size must be > 0".into(),
        ));
    }
    if args.sample_rate <= 0.0 {
        return Err(MagspecError::InvalidRemapArgs(
            "sample_rate must be > 0".into(),
        ));
    }
    if args.num_mel_bins < 1 {
        return Err(MagspecError::InvalidRemapArgs(
            "num_mel_bins must be >= 1".into(),
        ));
    }
    let f_max = args.f_max.unwrap_or(args.sample_rate / 2.0);
    if f_max <= args.f_min {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max ({f_max}) must be > f_min ({})",
            args.f_min
        )));
    }
    if f_max > args.sample_rate / 2.0 + 1e-4 {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max ({f_max}) exceeds Nyquist ({})",
            args.sample_rate / 2.0
        )));
    }
    let expected_freq_bins = args.fft_size / 2 + 1;
    if num_freq_bins != expected_freq_bins {
        return Err(MagspecError::InvalidFrame(format!(
            "frame height (freq bins) is {num_freq_bins} but fft_size/2+1 = {expected_freq_bins}"
        )));
    }

    let fb = build_filterbank_matrix(
        args.sample_rate,
        args.fft_size,
        num_freq_bins,
        args.num_mel_bins,
        args.f_min,
        f_max,
        args.norm,
        args.scale,
    )?;

    let mut out = try_vec![T::zero(); args.num_mel_bins * num_frames];

    for m in 0..args.num_mel_bins {
        let filter_row = &fb[m * num_freq_bins..(m + 1) * num_freq_bins];
        let out_row = &mut out[m * num_frames..(m + 1) * num_frames];

        for (k, &weight) in filter_row.iter().enumerate() {
            if weight == 0.0 {
                continue; // filterbank is sparse — skip zero-weight bins
            }
            let spec_row = &frame.data[k * num_frames..(k + 1) * num_frames];
            for (dst, &src) in out_row.iter_mut().zip(spec_row.iter()) {
                T::mel_accum(dst, weight, src);
            }
        }
    }

    Ok(StftFrameMut {
        data: BufferStoreMut::Owned(out),
        width: num_frames,
        height: args.num_mel_bins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_magnitude_frame(num_freq_bins: usize, num_frames: usize) -> Vec<f32> {
        // Flat spectrum: all ones
        vec![1.0f32; num_freq_bins * num_frames]
    }

    #[test]
    fn test_filterbank_shape() {
        let fft_size = 2048usize;
        let num_freq_bins = fft_size / 2 + 1;
        let num_frames = 10usize;
        let num_mel_bins = 128usize;

        let data = dummy_magnitude_frame(num_freq_bins, num_frames);
        let frame = StftFrame {
            data: std::borrow::Cow::Borrowed(&data),
            width: num_frames,
            height: num_freq_bins,
        };

        let args = MelFilterbankArgs {
            sample_rate: 22050.0,
            fft_size,
            num_mel_bins,
            f_min: 0.0,
            f_max: None,
            norm: MelNorm::Slaney,
            scale: MelScale::Htk,
        };

        let result = args.apply(&frame).unwrap();
        assert_eq!(result.height, num_mel_bins);
        assert_eq!(result.width, num_frames);
        assert_eq!(
            result.data.borrow().as_ref().len(),
            num_mel_bins * num_frames
        );
    }

    #[test]
    fn test_filterbank_nonnegative() {
        let fft_size = 1024usize;
        let num_freq_bins = fft_size / 2 + 1;
        let num_frames = 5usize;
        let data = dummy_magnitude_frame(num_freq_bins, num_frames);
        let frame = StftFrame {
            data: std::borrow::Cow::Borrowed(&data),
            width: num_frames,
            height: num_freq_bins,
        };
        let args = MelFilterbankArgs {
            sample_rate: 16000.0,
            fft_size,
            num_mel_bins: 40,
            f_min: 80.0,
            f_max: Some(7600.0),
            norm: MelNorm::None,
            scale: MelScale::Slaney,
        };
        let result = args.apply(&frame).unwrap();
        for &v in result.data.borrow().iter() {
            assert!(v >= 0.0, "negative value in mel output: {v}");
        }
    }

    #[test]
    fn test_htk_vs_slaney_differ() {
        // HTK and Slaney scales should produce different centre frequencies
        let scale = MelScale::Htk;
        let hz = 1000.0f32;
        let mel_htk = MelScale::Htk.hz_to_mel(hz);
        let mel_slaney = MelScale::Slaney.hz_to_mel(hz);
        // At 1 kHz the two scales diverge noticeably
        assert!(
            (mel_htk - mel_slaney).abs() > 0.5,
            "scales unexpectedly identical"
        );
        // Round-trip
        let rt = scale.mel_to_hz(scale.hz_to_mel(hz));
        assert!((rt - hz).abs() < 1e-3, "HTK round-trip failed: {rt}");
    }

    #[test]
    fn test_slaney_round_trip() {
        let scale = MelScale::Slaney;
        for &hz in &[50.0f32, 200.0, 500.0, 1000.0, 4000.0, 8000.0] {
            let rt = scale.mel_to_hz(scale.hz_to_mel(hz));
            assert!(
                (rt - hz).abs() < 1e-2,
                "Slaney round-trip failed at {hz} Hz: got {rt}"
            );
        }
    }

    #[test]
    fn test_build_filterbank_returns_correct_size() {
        let args = MelFilterbankArgs {
            sample_rate: 44100.0,
            fft_size: 2048,
            num_mel_bins: 80,
            f_min: 20.0,
            f_max: Some(20000.0),
            norm: MelNorm::Slaney,
            scale: MelScale::Htk,
        };
        let fb = args.build_filterbank().unwrap();
        assert_eq!(fb.len(), 80 * (2048 / 2 + 1));
    }
}
