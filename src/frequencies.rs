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
use crate::mla::{c_mul_t_add_fast, fmla};
use crate::{BufferStoreMut, MagspecError, StftFrame, StftFrameMut};
use num_complex::Complex;
use num_traits::Zero;

#[derive(Debug, Clone, Copy)]
pub enum FreqInterpMethod {
    /// Bilinear (linear) interpolation between adjacent frequency bins.
    Bilinear,
    /// Catmull-Rom cubic spline interpolation across 4 adjacent frequency bins.
    CatmullRom,
}

/// Arguments for log-scale frequency axis remapping.
#[derive(Debug, Clone, Copy)]
pub struct FreqRemapArgs {
    /// Sample rate of the original audio signal in Hz.
    pub sample_rate: f32,
    /// FFT size used to produce the STFT frame. Determines Hz-per-bin spacing
    /// as `sample_rate / fft_size`.
    pub fft_size: usize,
    /// Lowest output frequency in Hz. Must be > 0 and < `f_max`.
    /// Typically 20 Hz for audio, or `sample_rate / fft_size` to start at bin 1.
    pub f_min: f32,
    /// Highest output frequency in Hz. Must be <= `sample_rate / 2` (Nyquist).
    pub f_max: f32,
    /// Number of output frequency bins. Can differ from the input bin count —
    /// increase for more frequency resolution in the log domain, decrease to
    /// compress. Typically set equal to the input `height` or a fixed display height.
    pub num_bins_out: usize,
    /// Interpolation method to use along the frequency axis.
    pub method: FreqInterpMethod,
}

impl FreqRemapArgs {
    pub fn apply<'a>(
        &self,
        frame: &StftFrame<'a, Complex<f32>>,
    ) -> Result<StftFrameMut<'static, Complex<f32>>, MagspecError> {
        remap_freq_log_interp(frame, self)
    }
}

struct BinMap {
    lo: usize,
    hi: usize,
    alpha: f32,
}

/// Remap the frequency axis of an STFT frame from linear to log scale.
///
/// Each output bin is mapped to a log-spaced frequency between `f_min` and
/// `f_max`, then interpolated from the nearest input bins using the chosen
/// [`FreqInterpMethod`]. This compresses high frequencies and expands low
/// frequencies, matching how pitch is perceived by the human ear.
///
/// # Layout
/// Expects `data[freq_bin * num_frames + time_frame]`,
/// i.e. `width = num_frames`, `height = num_bins`.
/// Output follows the same layout with `width = num_bins_out`.
pub fn remap_freq_log_interp(
    frame: &StftFrame<'_, Complex<f32>>,
    args: &FreqRemapArgs,
) -> Result<StftFrameMut<'static, Complex<f32>>, MagspecError> {
    let num_frames = frame.width;
    let num_bins_in = frame.height;

    if num_frames == 0 || num_bins_in == 0 {
        return Err(MagspecError::InvalidFrame(format!(
            "frame dimensions are zero (width={}, height={})",
            num_frames, num_bins_in
        )));
    }
    if frame.data.len() != num_frames * num_bins_in {
        return Err(MagspecError::InvalidFrame(format!(
            "data length {} does not match width * height = {}",
            frame.data.len(),
            num_frames * num_bins_in
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
    if args.num_bins_out < 2 {
        return Err(MagspecError::InvalidRemapArgs(
            "num_bins_out must be >= 2".into(),
        ));
    }
    if args.f_min <= 0.0 {
        return Err(MagspecError::InvalidRemapArgs("f_min must be > 0".into()));
    }
    if args.f_max <= args.f_min {
        return Err(MagspecError::InvalidRemapArgs(format!(
            "f_max ({}) must be > f_min ({})",
            args.f_max, args.f_min
        )));
    }

    let bin_hz = args.sample_rate / args.fft_size as f32;

    if args.num_bins_out == num_bins_in
        && (args.f_min / bin_hz) < 0.001
        && (args.f_max / bin_hz - (num_bins_in - 1) as f32).abs() < 0.001
    {
        return Ok(StftFrameMut {
            data: BufferStoreMut::Owned(frame.data.to_vec()),
            width: frame.width,
            height: frame.height,
        });
    }

    let bin_maps: Vec<BinMap> = (0..args.num_bins_out)
        .map(|j| {
            let t = j as f32 / (args.num_bins_out - 1) as f32;
            let frac = (args.f_min * (args.f_max / args.f_min).powf(t) / bin_hz)
                .clamp(0.0, (num_bins_in - 1) as f32);
            let lo = frac.floor() as usize;
            BinMap {
                lo,
                hi: (lo + 1).min(num_bins_in - 1),
                alpha: frac.fract(),
            }
        })
        .collect();

    let mut out = try_vec![Complex::zero(); args.num_bins_out * num_frames];

    match args.method {
        FreqInterpMethod::Bilinear => {
            for (j, bm) in bin_maps.iter().enumerate() {
                let a1 = bm.alpha;
                let a0 = 1.0 - a1;

                let row_lo = &frame.data[bm.lo * num_frames..(bm.lo + 1) * num_frames];
                let row_hi = &frame.data[bm.hi * num_frames..(bm.hi + 1) * num_frames];
                let row_out = &mut out[j * num_frames..(j + 1) * num_frames];

                row_out
                    .iter_mut()
                    .zip(row_lo.iter().zip(row_hi.iter()))
                    .for_each(|(out, (lo, hi))| {
                        *out = c_mul_t_add_fast(*lo, a0, *hi * a1);
                    });
            }
        }

        FreqInterpMethod::CatmullRom => {
            for (j, bm) in bin_maps.iter().enumerate() {
                let i0 = bm.lo.saturating_sub(1);
                let i1 = bm.lo;
                let i2 = bm.hi;
                let i3 = (bm.lo + 2).min(num_bins_in - 1);

                let row0 = &frame.data[i0 * num_frames..(i0 + 1) * num_frames];
                let row1 = &frame.data[i1 * num_frames..(i1 + 1) * num_frames];
                let row2 = &frame.data[i2 * num_frames..(i2 + 1) * num_frames];
                let row3 = &frame.data[i3 * num_frames..(i3 + 1) * num_frames];
                let row_out = &mut out[j * num_frames..(j + 1) * num_frames];

                let a2 = bm.alpha * bm.alpha;
                let a3 = a2 * bm.alpha;
                let w0 = fmla(-0.5, a3, fmla(1.0, a2, -0.5 * bm.alpha));
                let w1 = fmla(1.5, a3, fmla(-2.5, a2, 1.0));
                let w2 = fmla(-1.5, a3, fmla(2.0, a2, 0.5 * bm.alpha));
                let w3 = fmla(0.5, a3, -0.5 * a2);

                row_out
                    .iter_mut()
                    .zip(
                        row0.iter()
                            .zip(row1.iter().zip(row2.iter().zip(row3.iter()))),
                    )
                    .for_each(|(out, (r0, (r1, (r2, r3))))| {
                        *out = c_mul_t_add_fast(
                            *r0,
                            w0,
                            c_mul_t_add_fast(*r1, w1, c_mul_t_add_fast(*r2, w2, *r3 * w3)),
                        );
                    });
            }
        }
    }

    Ok(StftFrameMut {
        data: BufferStoreMut::Owned(out),
        width: args.num_bins_out,
        height: num_frames,
    })
}
