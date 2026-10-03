#!/usr/bin/env python
# coding: utf-8

# # Exercise 2 

# The code implements a lossless audio compression system using Rice coding with an adaptive prediction approach. It reads WAV audio files, analyzes the signal, and decides whether prediction will help compression. The program tests three options: no prediction, first-order delta prediction, and second-order prediction. It compares their standard deviations and automatically selects the method that produces the smallest residual values. These residuals are then encoded using Rice coding and written to a compressed file. The program also includes a decompression step that reconstructs the original audio and verifies that the decoded file matches the original. The outputs show how different signals respond to compression depending on their characteristics.

# # Imports + setup

# In[1]:


# ============================================================================
# EXERCISE 2 - Rice Coding
# Imports + setup
# ============================================================================

import numpy as np
import wave
import struct
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

print("✅ Libraries imported successfully!")


# # Bit stream tools

# In[2]:


# ============================================================================
# SECTION 1: Memory-Efficient Bit Stream Handlers
# ============================================================================

class BitWriter:
    """Write bits directly to file."""
    
    def __init__(self, file_handle):
        self.file = file_handle
        self.buffer = 0
        self.buffer_size = 0
        
    def write_bit(self, bit):
        # Add bit into buffer
        self.buffer = (self.buffer << 1) | bit
        self.buffer_size += 1
        
        # Write byte when full
        if self.buffer_size == 8:
            self.file.write(bytes([self.buffer]))
            self.buffer = 0
            self.buffer_size = 0
    
    def write_bits(self, value, num_bits):
        # Write value bit-by-bit
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)
    
    def flush(self):
        # Pad remaining bits to full byte
        if self.buffer_size > 0:
            padding = 8 - self.buffer_size
            self.buffer = self.buffer << padding
            self.file.write(bytes([self.buffer]))
            return padding
        return 0


class BitReader:
    """Read bits directly from file."""
    
    def __init__(self, file_handle):
        self.file = file_handle
        self.buffer = 0
        self.buffer_size = 0
        
    def read_bit(self):
        # Load new byte if needed
        if self.buffer_size == 0:
            byte_data = self.file.read(1)
            if not byte_data:
                raise EOFError("End of file")
            self.buffer = byte_data[0]
            self.buffer_size = 8
        
        bit = (self.buffer >> 7) & 1
        self.buffer = (self.buffer << 1) & 0xFF
        self.buffer_size -= 1
        return bit
    
    def read_bits(self, num_bits):
        # Read N bits as int
        value = 0
        for _ in range(num_bits):
            value = (value << 1) | self.read_bit()
        return value


# # Rice coding core

# In[3]:


# ============================================================================
# SECTION 2: Rice Coding Algorithm
# ============================================================================

def zigzag_encode(value):
    """Signed -> unsigned mapping."""
    return (2 * value) if value >= 0 else (-2 * value - 1)


def zigzag_decode(value):
    """Unsigned -> signed mapping."""
    return (value // 2) if (value % 2 == 0) else (-(value + 1) // 2)


def encode_rice(writer, value, k):
    """Encode one value using Rice."""
    mapped = zigzag_encode(value)
    quotient = mapped >> k
    remainder = mapped & ((1 << k) - 1)
    
    # Unary for quotient
    for _ in range(quotient):
        writer.write_bit(0)
    writer.write_bit(1)
    
    # Binary for remainder
    writer.write_bits(remainder, k)


def decode_rice(reader, k):
    """Decode one Rice value."""
    quotient = 0
    while reader.read_bit() == 0:
        quotient += 1
    
    remainder = reader.read_bits(k)
    mapped = (quotient << k) | remainder
    return zigzag_decode(mapped)


# # Prediction methods

# In[4]:


# ============================================================================
# SECTION 3: Prediction Strategies
# ============================================================================

def apply_no_prediction(audio_data):
    """Store raw samples (no prediction)."""
    return audio_data.copy()


def reverse_no_prediction(data):
    """Nothing to reverse."""
    return data.copy()


def apply_first_order_delta(audio_data):
    """Delta: x[n] - x[n-1]."""
    deltas = np.zeros(len(audio_data), dtype=np.int32)
    deltas[0] = audio_data[0]
    deltas[1:] = np.diff(audio_data.astype(np.int32))
    return deltas.astype(np.int16)


def reverse_first_order_delta(deltas):
    """Undo delta by cumulative sum."""
    return np.cumsum(deltas, dtype=np.int16)


def apply_second_order_prediction(audio_data):
    """2nd order residuals."""
    residuals = np.zeros(len(audio_data), dtype=np.int32)
    residuals[0] = audio_data[0]
    if len(audio_data) > 1:
        residuals[1] = audio_data[1]
    
    # Predict using last 2 samples
    for i in range(2, len(audio_data)):
        predicted = 2 * int(audio_data[i-1]) - int(audio_data[i-2])
        residuals[i] = int(audio_data[i]) - predicted
    
    return residuals.astype(np.int16)


def reverse_second_order_prediction(residuals):
    """Rebuild original samples."""
    audio_data = np.zeros(len(residuals), dtype=np.int32)
    audio_data[0] = residuals[0]
    if len(residuals) > 1:
        audio_data[1] = residuals[1]
    
    for i in range(2, len(residuals)):
        predicted = 2 * audio_data[i-1] - audio_data[i-2]
        audio_data[i] = residuals[i] + predicted
    
    return audio_data.astype(np.int16)


# # Smart predictor choice

# In[5]:


# ============================================================================
# SECTION 4: SMART Encoding Selection
# ============================================================================

def choose_best_encoding(audio_data, k):
    """
    Choose best predictor using std dev test.
    0 = none, 1 = first order, 2 = second order
    """
    sample_size = min(10000, len(audio_data))
    sample = audio_data[:sample_size]
    
    std_original = np.std(sample.astype(np.float32))
    
    # Try 1st order
    delta1 = apply_first_order_delta(sample)
    std_delta1 = np.std(delta1.astype(np.float32))
    
    # Try 2nd order
    delta2 = apply_second_order_prediction(sample)
    std_delta2 = np.std(delta2.astype(np.float32))
    
    print(f"\n   🔍 Signal Analysis:")
    print(f"      Original    - Std: {std_original:>10.2f}")
    print(f"      1st order   - Std: {std_delta1:>10.2f} (ratio: {std_delta1/std_original:.3f})")
    print(f"      2nd order   - Std: {std_delta2:>10.2f} (ratio: {std_delta2/std_original:.3f})")
    
    # Must improve by at least 5%
    IMPROVEMENT_THRESHOLD = 0.95
    
    if std_delta1 > std_original * IMPROVEMENT_THRESHOLD and        std_delta2 > std_original * IMPROVEMENT_THRESHOLD:
        print(f"      ⚠️  Prediction ineffective (noise/random signal)")
        print(f"      ✓ Selected: NO PREDICTION (raw encoding)")
        return apply_no_prediction(audio_data), 0
    
    elif std_delta2 < std_delta1 * 0.9:
        print(f"      ✓ Selected: 2nd order prediction")
        return apply_second_order_prediction(audio_data), 2
    
    else:
        print(f"      ✓ Selected: 1st order delta")
        return apply_first_order_delta(audio_data), 1


# # Residuals plot

# In[6]:


# ============================================================================
# SECTION 5: Visualization
# ============================================================================

def plot_residual_distribution(residuals, filename, k, encoding_type, output_dir):
    """Histogram + stats text."""
    plt.figure(figsize=(12, 5))
    
    # Histogram plot
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=200, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    type_names = {0: "No Prediction", 1: "1st Order Delta", 2: "2nd Order Prediction"}
    plt.title(f'{filename} - Distribution (K={k}, {type_names.get(encoding_type, "Unknown")})')
    plt.grid(True, alpha=0.3)
    
    # Stats panel
    plt.subplot(1, 2, 2)
    stats_text = f"""
    {filename} - K={k}
    Encoding: {type_names.get(encoding_type, "Unknown")}
    
    Samples: {len(residuals):,}
    Min: {np.min(residuals):,}
    Max: {np.max(residuals):,}
    Mean: {np.mean(residuals):.2f}
    Std Dev: {np.std(residuals):.2f}
    
    Distribution:
    [-10, 10]:      {np.sum(np.abs(residuals) <= 10) / len(residuals) * 100:>5.1f}%
    [-100, 100]:    {np.sum(np.abs(residuals) <= 100) / len(residuals) * 100:>5.1f}%
    [-1000, 1000]:  {np.sum(np.abs(residuals) <= 1000) / len(residuals) * 100:>5.1f}%
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_K{k}_residuals.png')
    plt.savefig(output_path, dpi=100)
    plt.show()
    
    print(f"   📊 Plot saved to: {output_path}")


# # Compression step

# In[7]:


# ============================================================================
# SECTION 6: Compression Function
# ============================================================================

def compress_audio(input_wav, output_file, k, show_plot=True, output_dir="Exercise2files"):
    """Compress WAV using Rice coding."""
    print(f"\n{'='*60}")
    print(f"🔄 ENCODING: {os.path.basename(input_wav)} (K={k})")
    print(f"{'='*60}")
    
    # Read audio
    sample_rate, audio_data = wavfile.read(input_wav)
    with wave.open(input_wav, 'rb') as wf:
        params = wf.getparams()
    
    original_size = os.path.getsize(input_wav)
    print(f"📄 Original: {original_size:,} bytes")
    print(f"   Rate: {sample_rate} Hz, Channels: {params.nchannels}")
    print(f"   Samples: {params.nframes:,}")
    
    # Handle stereo by flattening
    shape = audio_data.shape
    is_stereo = len(shape) > 1
    audio_flat = audio_data.flatten() if is_stereo else audio_data
    
    # Choose best predictor
    residuals, encoding_type = choose_best_encoding(audio_flat, k)
    
    # Print quick stats
    print(f"\n   Encoded data range: [{np.min(residuals)}, {np.max(residuals)}]")
    print(f"   Encoded data std dev: {np.std(residuals):.2f}")
    print(f"   Values in [-100, 100]: {np.sum(np.abs(residuals) <= 100) / len(residuals) * 100:.1f}%")
    
    # Optional plot
    if show_plot:
        plot_residual_distribution(residuals, os.path.basename(input_wav), k, encoding_type, output_dir)
    
    print(f"\n🔧 Rice Encoding {len(residuals):,} samples...")
    
    with open(output_file, 'wb') as f:
        # Header stores metadata for decoding
        header = struct.pack('IIIHHIIIHH',
                            sample_rate,
                            len(audio_flat),
                            params.sampwidth,
                            params.nchannels,
                            k,
                            shape[0],
                            shape[1] if is_stereo else 0,
                            0,              # padding placeholder
                            encoding_type,  # predictor type
                            0)
        f.write(header)
        
        writer = BitWriter(f)
        progress_step = max(1, len(residuals) // 20)
        
        for i, val in enumerate(residuals):
            encode_rice(writer, int(val), k)
            if (i + 1) % progress_step == 0:
                pct = (i + 1) / len(residuals) * 100
                print(f"   Progress: {pct:.0f}%", end='\r')
        
        print(f"   Progress: 100% ✓         ")
        
        # Flush last bits and store padding
        padding = writer.flush()
        f.seek(24)
        f.write(struct.pack('I', padding))
    
    compressed_size = os.path.getsize(output_file)
    
    # Print size results
    if compressed_size < original_size:
        ratio = (1 - compressed_size / original_size) * 100
        saved = original_size - compressed_size
        print(f"\n📊 Result: {compressed_size:,} bytes")
        print(f"   ✅ COMPRESSED by {saved:,} bytes ({ratio:.2f}%)")
        print(f"   Compression factor: {original_size/compressed_size:.2f}x")
    else:
        ratio = ((compressed_size / original_size) - 1) * 100
        extra = compressed_size - original_size
        print(f"\n📊 Result: {compressed_size:,} bytes")
        
        if ratio < 10:
            print(f"   ≈ Near original size (+{ratio:.2f}%)")
            print(f"   ℹ️  Signal not compressible with this K value")
        else:
            print(f"   ⚠️ EXPANDED by {extra:,} bytes (+{ratio:.2f}%)")
            print(f"   K={k} not suitable for this signal")
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': (1 - compressed_size / original_size) * 100,
        'encoding_type': encoding_type
    }


# # Decompression step

# In[8]:


# ============================================================================
# SECTION 7: Decompression Function
# ============================================================================

def decompress_audio(input_file, output_wav):
    """Decode .ex2 back into WAV."""
    print(f"\n{'='*60}")
    print(f"🔄 DECODING: {os.path.basename(input_file)}")
    print(f"{'='*60}")
    
    with open(input_file, 'rb') as f:
        # Read file header
        header = f.read(32)
        (sample_rate, num_samples, sampwidth, nchannels, k,
         shape0, shape1, padding, encoding_type, reserved) = struct.unpack('IIIHHIIIHH', header)
        
        type_names = {0: "None", 1: "1st order", 2: "2nd order"}
        print(f"📋 Metadata: {num_samples:,} samples, K={k}")
        print(f"   Prediction type: {type_names.get(encoding_type, 'Unknown')}")
        
        reader = BitReader(f)
        decoded = []
        progress_step = max(1, num_samples // 20)
        
        print(f"\n🔧 Rice Decoding...")
        for i in range(num_samples):
            val = decode_rice(reader, k)
            decoded.append(val)
            if (i + 1) % progress_step == 0:
                pct = (i + 1) / num_samples * 100
                print(f"   Progress: {pct:.0f}%", end='\r')
        
        print(f"   Progress: 100% ✓         ")
    
    decoded_array = np.array(decoded, dtype=np.int16)
    
    # Reverse prediction
    if encoding_type == 2:
        print(f"🔧 Reversing 2nd order prediction...")
        audio_flat = reverse_second_order_prediction(decoded_array)
    elif encoding_type == 1:
        print(f"🔧 Reversing 1st order delta...")
        audio_flat = reverse_first_order_delta(decoded_array)
    else:
        print(f"🔧 No prediction reversal needed...")
        audio_flat = reverse_no_prediction(decoded_array)
    
    # Reshape stereo if needed
    if shape1 > 0:
        audio_data = audio_flat.reshape(shape0, shape1)
    else:
        audio_data = audio_flat
    
    wavfile.write(output_wav, sample_rate, audio_data)
    print(f"\n✅ Decoded: {output_wav} ({os.path.getsize(output_wav):,} bytes)")


# # Run all files

# In[9]:


# ============================================================================
# SECTION 8: Main Processing
# ============================================================================

def process_sound_files():
    """Run Sound1 and Sound2 with K=4 and K=2."""
    print("\n" + "="*60)
    print("🎵 RICE CODING - SMART ADAPTIVE VERSION")
    print("="*60)
    
    BASE_PATH = "Exercise2files"
    sound_files = [
        os.path.join(BASE_PATH, "Sound1.wav"),
        os.path.join(BASE_PATH, "Sound2.wav")
    ]
    
    k_values = [4, 2]
    results = {}
    
    for sound_file in sound_files:
        if not os.path.exists(sound_file):
            print(f"\n⚠️ {sound_file} not found!")
            continue
        
        sound_name = os.path.basename(sound_file)
        results[sound_name] = {}
        
        print(f"\n\n{'#'*60}")
        print(f"# {sound_name}")
        print(f"{'#'*60}")
        
        for k in k_values:
            base = os.path.splitext(sound_name)[0]
            encoded_file = os.path.join(BASE_PATH, f"{base}_Enc_K{k}.ex2")
            decoded_file = os.path.join(BASE_PATH, f"{base}_Enc_K{k}_Dec.wav")
            
            # Compress then decode
            stats = compress_audio(sound_file, encoded_file, k, show_plot=(k==4), output_dir=BASE_PATH)
            decompress_audio(encoded_file, decoded_file)
            
            # Check lossless output
            print(f"\n🔍 Verification...")
            orig_rate, orig_data = wavfile.read(sound_file)
            dec_rate, dec_data = wavfile.read(decoded_file)
            
            if np.array_equal(orig_data, dec_data):
                print(f"   ✅ LOSSLESS - Files identical")
            else:
                max_diff = np.max(np.abs(orig_data.astype(np.int32) - dec_data.astype(np.int32)))
                print(f"   ❌ Error! Max diff: {max_diff}")
            
            results[sound_name][f'K={k}'] = stats
    
    # Final table
    print(f"\n\n{'='*100}")
    print("📊 FINAL COMPRESSION RESULTS")
    print(f"{'='*100}")
    print(f"\n{'File':<15} {'Original':<15} {'K=4 Size':<15} {'K=2 Size':<15} {'% K=4':<10} {'% K=2':<10} {'Method'}")
    print("-"*100)
    
    for sound_name in results:
        if 'K=4' in results[sound_name] and 'K=2' in results[sound_name]:
            orig = results[sound_name]['K=4']['original_size']
            k4 = results[sound_name]['K=4']['compressed_size']
            k2 = results[sound_name]['K=2']['compressed_size']
            r4 = results[sound_name]['K=4']['compression_ratio']
            r2 = results[sound_name]['K=2']['compression_ratio']
            t4 = results[sound_name]['K=4']['encoding_type']
            t2 = results[sound_name]['K=2']['encoding_type']
            
            type_str = f"{t4}/{t2}"
            print(f"{sound_name:<15} {orig:>13,} B {k4:>13,} B {k2:>13,} B {r4:>8.2f}% {r2:>8.2f}%  {type_str:>6}")
    
    print("\n" + "="*100)
    print("\n✅ Processing complete!")
    print("\nEncoding types: 0=No prediction, 1=1st order delta, 2=2nd order prediction")
    print(f"\nAll files saved in: {BASE_PATH}/")
    
    return results


# # Run + outputs

# In[10]:


# ============================================================================
# RUN
# ============================================================================

results = process_sound_files()

print("\n" + "="*60)
print("📁 Output Files in Exercise2files/:")
print("="*60)
print("  Encoded:")
print("    - Sound1_Enc_K4.ex2")
print("    - Sound1_Enc_K2.ex2")
print("    - Sound2_Enc_K4.ex2")
print("    - Sound2_Enc_K2.ex2")
print("\n  Decoded:")
print("    - Sound1_Enc_K4_Dec.wav")
print("    - Sound1_Enc_K2_Dec.wav")
print("    - Sound2_Enc_K4_Dec.wav")
print("    - Sound2_Enc_K2_Dec.wav")
print("\n  Plots:")
print("    - Sound1_K4_residuals.png")
print("    - Sound2_K4_residuals.png")

