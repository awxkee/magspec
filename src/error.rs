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
use std::fmt::Formatter;

#[derive(Debug)]
pub enum MagspecError {
    Allocation(usize),
    FftError(String),
    InvalidScratchSize(usize, usize),
    InvalidFrame(String),
    InvalidRemapArgs(String),
    FreqOutOfRange { f: f32, min: f32, max: f32 },
}

impl std::fmt::Display for MagspecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MagspecError::FftError(msg) => write!(f, "Internal FFT error: {}", msg),
            MagspecError::InvalidScratchSize(size, expected) => {
                write!(f, "Scratch size is {} but expected {}", size, expected)
            }
            MagspecError::Allocation(size) => {
                write!(f, "Failed to allocate buffer of size {}", size)
            }
            MagspecError::InvalidFrame(s) => write!(f, "Invalid frame: {}", s),
            MagspecError::InvalidRemapArgs(s) => write!(f, "Invalid remap arguments: {}", s),
            MagspecError::FreqOutOfRange { f: freq, min, max } => write!(
                f,
                "Frequency {:.2} Hz is out of range [{:.2}, {:.2}] Hz",
                freq, min, max
            ),
        }
    }
}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::error::MagspecError::Allocation($n))?;
        v.resize($n, $elem);
        v
    }};
}

pub(crate) use try_vec;
