//! Image preprocessing: load, resize to model input size.
//!
//! Uses plain bilinear sampling (matching OpenCV `INTER_LINEAR`) rather than
//! `image::imageops::resize`, whose Triangle filter applies anti-aliasing on
//! downscale and produces different pixels than the Python pipeline.

use image::GenericImageView;

use crate::error::Result;

/// Plain bilinear resize of an RGB8 buffer — bit-compatible with
/// `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
fn resize_bilinear_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;

    for y in 0..dst_h {
        let fy = ((y as f64 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f64;

        for x in 0..dst_w {
            let fx = ((x as f64 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f64;

            let p00 = (y0 * src_w + x0) * 3;
            let p01 = (y0 * src_w + x1) * 3;
            let p10 = (y1 * src_w + x0) * 3;
            let p11 = (y1 * src_w + x1) * 3;
            let d = (y * dst_w + x) * 3;

            for c in 0..3 {
                let v = src[p00 + c] as f64 * (1.0 - wx) * (1.0 - wy)
                    + src[p01 + c] as f64 * wx * (1.0 - wy)
                    + src[p10 + c] as f64 * (1.0 - wx) * wy
                    + src[p11 + c] as f64 * wx * wy;
                dst[d + c] = (v + 0.5) as u8;
            }
        }
    }
    dst
}

/// Load an image and resize it to the model's input dimensions.
///
/// Returns a flat RGB buffer (NHWC layout) suitable for RKNN input.
pub fn preprocess(img_path: &str, target_w: usize, target_h: usize) -> Result<Vec<u8>> {
    let img = image::open(img_path)?;
    let (orig_w, orig_h) = img.dimensions();
    log::info!(
        "Preprocess: {}x{} -> {}x{} (bilinear)",
        orig_w,
        orig_h,
        target_w,
        target_h
    );

    let rgb = img.to_rgb8();
    Ok(resize_bilinear_rgb(
        rgb.as_raw(),
        orig_w as usize,
        orig_h as usize,
        target_w,
        target_h,
    ))
}

/// Load an image and return it as a DynamicImage (for point cloud color lookup).
pub fn load_image(img_path: &str) -> Result<image::DynamicImage> {
    Ok(image::open(img_path)?)
}
