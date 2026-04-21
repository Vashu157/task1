import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import io

st.set_page_config(page_title="Image Processor", layout="wide")

st.title("Image Compression & Enhancement")
st.write("Upload an image. We will compress it, then attempt to increase its size and enhance it, comparing the final result with the original.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = original_image.size
    
    st.markdown("### 1. Original Image")
    st.image(original_image, caption=f"Original ({orig_w}x{orig_h})", use_container_width=True)
    
    st.sidebar.header("Processing Parameters")
    quality = st.sidebar.slider("Compression Quality", 1, 100, 10)
    scale = st.sidebar.slider("Downscale Factor", 2, 10, 4)
    
    # Compress and resize down
    down_w = max(1, orig_w // scale)
    down_h = max(1, orig_h // scale)
    
    small_image = original_image.resize((down_w, down_h), Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    small_image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer).convert("RGB")
    
    st.markdown("---")
    st.markdown("### 2. Compressed Image")
    st.image(compressed_image, caption=f"Compressed and Downscaled ({down_w}x{down_h}, Q={quality})", use_container_width=True)
    
    # 3. Retrieve and Restore Features (Enhancement)
    # Upscale back to original resolution using Lanczos (high quality to preserve details)
    upscaled_image = compressed_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    # Feature Restoration Pipeline:
    # 1. Smooth out compression blockiness
    smoothed_image = upscaled_image.filter(ImageFilter.SMOOTH_MORE)
    # 2. Unsharp Masking to selectively restore edges and high-frequency details
    improved_image = smoothed_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    enhanced_buffer = io.BytesIO()
    improved_image.save(enhanced_buffer, format="JPEG", quality=95)
    enh_size_kb = len(enhanced_buffer.getvalue()) / 1024
    
    st.markdown("---")
    st.markdown("### 3. Improved Image (Restored Features)")
    st.image(improved_image, caption=f"Restored & Upscaled ({orig_w}x{orig_h}) | Estimated Final Size: {enh_size_kb:.1f} KB", use_container_width=True)
    
    # Calculate metrics
    orig_np = np.array(original_image)
    impr_np = np.array(improved_image)
    
    st.markdown("---")
    st.markdown("### 4. Metrics Comparison (Original vs Improved)")
    
    m_val = mse(orig_np, impr_np)
    p_val = psnr(orig_np, impr_np, data_range=255)
    
    win_size = min(7, orig_np.shape[0]-1, orig_np.shape[1]-1)
    if win_size % 2 == 0:
        win_size -= 1
        
    if win_size >= 3:
        s_val = ssim(orig_np, impr_np, multichannel=True, channel_axis=-1, data_range=255, win_size=win_size)
        ssim_text = f"{s_val:.4f}"
        
        # Naive Comparison
        quality_lost = (1.0 - s_val) * 100
        quality_retained = s_val * 100
        
        orig_size_kb = uploaded_file.size / 1024
        comp_size_kb = len(buffer.getvalue()) / 1024
        size_saved = ((orig_size_kb - comp_size_kb) / orig_size_kb) * 100 if orig_size_kb > 0 else 0
        
        st.info(f"**Simple Breakdown:**\n\n"
                f"- **The original image is roughly {quality_lost:.1f}% better** in terms of structural detail and sharpness.\n"
                f"- By restoring features, we managed to reproduce about **{quality_retained:.1f}%** of the original image quality in the final version.\n"
                f"- **File Size Journey:** Original ({orig_size_kb:.1f} KB) → Compressed ({comp_size_kb:.1f} KB) → Final Enhanced Image ({enh_size_kb:.1f} KB).")
    else:
        ssim_text = "N/A (Image too small)"

    st.markdown("**Technical Metrics:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error (MSE)", f"{m_val:.2f}", delta="Lower is better", delta_color="inverse")
    col2.metric("PSNR (dB)", f"{p_val:.2f}", delta="Higher is better")
    col3.metric("Structural Similarity (SSIM)", ssim_text, delta="Closer to 1 is better")
