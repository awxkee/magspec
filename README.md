# Magspec

`magspec` is a **Short-Time Fourier Transform (STFT)** library written in Rust, designed for
speed, safety, and high-quality spectral analysis.

It provides fast STFT execution for `f32` and `f64`, flexible window types, customizable hop sizes,
and utilities for generating magnitude- or power-based spectrograms.

# Example

```rust
let stft_plan = Magspec::make_forward_f32(StftOptions {
    len: 256,
    hop_size: 32,
    window: StftWindow::Blackman,
    normalize: true,
})
.unwrap();
let stft_result = stft_plan.execute(&signal)?;
// optionally draw spectrogram using spectrograph
use spectrograph::{Normalizer, SpectrographOptions, rgb_spectrograph_color_f32, SpectrographFrame};
let scalogram = rgb_spectrograph_f32(&SpectrographFrame {
    data: std::borrow::Cow::Borrowed(stft_result.data.borrow()),
    width: stft_result.width,
    height: stft_result.height,
    }, SpectrographOptions {
    out_width: 1920,
    out_height: 1080,
    normalizer: Normalizer::LogMagnitude,
    colormap: spectrograph::Colormap::Turbo,
})?;
```

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.