# Lossless Audio Compression

A complete lossless audio compression system written in Python. It compresses WAV files with **Rice coding** and an **adaptive predictor** that picks the best way to prepare each signal, then decompresses the result and checks that every sample matches the original.

## How It Works

1. **Read** a WAV file (mono or stereo, 16-bit).
2. **Analyse the signal.** The first 10,000 samples are tested with three options, and the one that makes the values smallest is chosen. A predictor must reduce the standard deviation by at least 5% to be used:

   | Type | Method | Residual |
   |------|--------|----------|
   | 0 | No prediction | the raw samples |
   | 1 | First-order delta | `x[n] - x[n-1]` |
   | 2 | Second-order prediction | `x[n] - (2·x[n-1] - x[n-2])` |

3. **Encode** the residuals with Rice coding:
   - Each signed value is mapped to an unsigned one with zigzag encoding.
   - The value is split with parameter `K` into a quotient, stored in unary, and a remainder, stored in `K` bits.
   - Bits are written straight to the file with a bit-stream writer, so the whole signal never has to sit in memory as bits.
4. **Write the `.ex2` file** with a 32-byte header (sample rate, sample count, sample width, channels, `K`, shape, padding and predictor type) followed by the Rice-coded data.
5. **Decode.** The decoder reads the header, Rice-decodes the values, reverses the prediction and writes a WAV file.
6. **Verify** that the decoded samples are identical to the original (`np.array_equal`).
7. **Plot** the distribution of the residuals (histogram and statistics) for each file.

## Results

The program compresses two test sounds with `K=4` and `K=2`. Both decode back to a **bit-identical** copy of the original in every case.

| Sound | Original | Predictor chosen | K=4 size | K=2 size |
|-------|----------|------------------|----------|----------|
| Sound1 | 1,002,088 B | 2nd order (std ratio 0.374) | 706,524 B (**29.5% smaller**) | 1,828,619 B (82.5% larger) |
| Sound2 | 1,008,044 B | None (std ratio above 1) | 39,103,001 B (38.8x larger) | 155,435,741 B (154.2x larger) |

What this shows:

- **Sound1** is smooth, so prediction cuts the signal's standard deviation from 58.6 to 21.9. Rice coding with `K=4` then shrinks the file by about 30%.
- **Sound2** is loud and noise-like (standard deviation about 10,700), and both predictors make it worse, so the program correctly picks no prediction. Such large values suit a large `K`. With `K=2` or `K=4` the unary part of each code becomes very long, and the file grows instead of shrinking.
- The choice of `K` matters. Rice coding works best when `K` matches the size of the values being coded.

## Project Structure

| File / folder | Purpose |
|---------------|---------|
| `Exercise 2.py` | The whole system: bit-stream tools, Rice coding, predictors, compression, decompression and the test run |
| `Exercise2files/Sound1.wav`, `Sound2.wav` | The two input sounds |
| `Exercise2files/Sound1_Enc_K4.ex2`, `Sound1_Enc_K2.ex2` | Compressed output files for Sound1 |
| `Exercise2files/*_Dec.wav` | The decoded WAV files, identical to the originals |
| `Exercise2files/*_residuals.png` | Residual distribution plots |

The compressed files for Sound2 are about 39 MB and 155 MB, so they are not stored here. Running the script recreates them.

## Run Locally

You need Python 3 with NumPy, SciPy and Matplotlib.

```bash
git clone https://github.com/ayeshaatif07/Lossless-Audio-Compression.git
cd Lossless-Audio-Compression
pip install numpy scipy matplotlib
python3 "Exercise 2.py"
```

The script compresses and decodes both sounds with `K=4` and `K=2`, prints the signal analysis and a results table, and saves the plots. It is written in pure Python, so encoding and decoding take a while, especially for Sound2. Run it from the project folder, because it reads the sounds from `Exercise2files/`.

## Author and Credits

- **Author:** Ayesha Atif
- **Coursework:** Intelligent Signal Processing
- Built with Python, [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/).
