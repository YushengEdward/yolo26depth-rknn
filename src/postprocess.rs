//! Depth post-processing: resize to original image size and colormap generation.

use image::RgbImage;
use log::info;

use crate::cli::ColorMode;
use crate::error::Result;

// ============================================================================
// Bilinear resize for f32 depth maps
// ============================================================================

/// Resize a depth map from model output size to the original image size.
pub fn resize_depth(
    depth_raw: &[f32],
    model_w: usize,
    model_h: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<f32> {
    if model_w == target_w && model_h == target_h {
        return depth_raw.to_vec();
    }

    info!(
        "Resize depth: {}x{} -> {}x{} (bilinear)",
        model_w, model_h, target_w, target_h
    );

    let mut dst = vec![0.0f32; target_w * target_h];
    let sx = (model_w as f64) / (target_w as f64);
    let sy = (model_h as f64) / (target_h as f64);

    for y in 0..target_h {
        let sy_f = (y as f64 + 0.5) * sy - 0.5;
        let y0 = sy_f.floor() as usize;
        let y1 = (y0 + 1).min(model_h - 1);
        let fy = sy_f - y0 as f64;

        for x in 0..target_w {
            let sx_f = (x as f64 + 0.5) * sx - 0.5;
            let x0 = sx_f.floor() as usize;
            let x1 = (x0 + 1).min(model_w - 1);
            let fx = sx_f - x0 as f64;

            let v = depth_raw[y0 * model_w + x0] as f64 * (1.0 - fx) * (1.0 - fy)
                + depth_raw[y0 * model_w + x1] as f64 * fx * (1.0 - fy)
                + depth_raw[y1 * model_w + x0] as f64 * (1.0 - fx) * fy
                + depth_raw[y1 * model_w + x1] as f64 * fx * fy;
            dst[y * target_w + x] = v as f32;
        }
    }
    dst
}

// ============================================================================
// Statistics helpers
// ============================================================================

/// Linear interpolation percentile from a sorted slice.
pub fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Print depth statistics to stderr via log.
pub fn print_stats(depth: &[f32]) {
    let valid: Vec<f32> = depth.iter().copied().filter(|&d| d > 0.0).collect();
    if valid.is_empty() {
        log::warn!("No valid depth values!");
        return;
    }
    let mut sorted = valid;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let p5 = percentile(&sorted, 5.0);
    let p95 = percentile(&sorted, 95.0);
    info!("Depth median: {:.2} m", median);
    info!("Depth 5%-95%: {:.1} ~ {:.1} m", p5, p95);
}

// ============================================================================
// JET colormap
// ============================================================================

/// OpenCV-compatible JET colormap (`cv2.COLORMAP_JET`). Returns [R, G, B].
///
/// Each channel is a trapezoid: `c = clamp(1.5 - |4t - k|, 0, 1)`
/// with k = 3, 2, 1 for R, G, B. Endpoints are dark blue / dark red.
fn jet_color(t: f32) -> [u8; 3] {
    let r = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let g = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let b = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    [
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    ]
}

// ============================================================================
// Colormap generation
// ============================================================================

/// Convert a depth map to a JET colormap image and save it.
pub fn save_colormap(
    depth: &[f32],
    width: usize,
    height: usize,
    output_path: &str,
    mode: ColorMode,
) -> Result<()> {
    let pixels: Vec<u8> = match mode {
        ColorMode::Disparity => disparity_to_pixels(depth),
        ColorMode::Metric => metric_to_pixels(depth),
    };

    let img = RgbImage::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| crate::error::Error::Invalid("failed to create image buffer".into()))?;
    img.save(output_path)?;
    info!("Colormap saved: {}", output_path);
    Ok(())
}

/// Disparity mode: 1/d with 2%-98% percentile clipping.
fn disparity_to_pixels(depth: &[f32]) -> Vec<u8> {
    let n = depth.len();
    let mut valid_disparity = Vec::new();
    let mut disparity = vec![0.0f32; n];

    for (i, &d) in depth.iter().enumerate() {
        if d > 0.0 {
            let disp = 1.0 / d;
            disparity[i] = disp;
            valid_disparity.push(disp);
        }
    }

    valid_disparity.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile(&valid_disparity, 2.0);
    let hi = percentile(&valid_disparity, 98.0);
    let range = (hi - lo).max(1e-6);

    depth
        .iter()
        .zip(disparity.iter())
        .flat_map(|(&d, &v)| {
            if d <= 0.0 {
                [0u8, 0u8, 0u8]
            } else {
                let t = ((v - lo) / range).clamp(0.0, 1.0);
                jet_color(t)
            }
        })
        .collect()
}

/// Metric mode: linear mapping over the valid depth range.
fn metric_to_pixels(depth: &[f32]) -> Vec<u8> {
    let min_d = depth.iter().copied().fold(f32::INFINITY, f32::min);
    let max_d = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_d - min_d).max(1e-8);

    depth
        .iter()
        .flat_map(|&v| {
            let t = ((v - min_d) / range).clamp(0.0, 1.0);
            jet_color(t)
        })
        .collect()
}
