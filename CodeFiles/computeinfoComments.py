# ══════════════════════════════════════════════════════════════════════════════
#  compute_info_comments.py
#  Reference notes for every feature computed in computeinfo.py.
#  Explains what each feature measures and what high/low values indicate
#  in terms of AI-generated vs human-generated music.
# ══════════════════════════════════════════════════════════════════════════════


# ── RMS (Root Mean Square) ────────────────────────────────────────────────────
# A measure of the overall energy of the signal — a higher RMS can indicate
# a louder sound, while a lower RMS can indicate a quieter sound.
# AI music might have different RMS patterns than human music, depending on
# how it was generated.
# Sectional RMS (first 5s, first 30s, middle 20s, last 20s) can reveal
# whether the track has a "perfect" or "consistent" start vs a more gradual
# human build-up. AI music might have a more uniform RMS across sections.


# ── ZCR (Zero-Crossing Rate) ──────────────────────────────────────────────────
# How often the signal crosses zero — a higher ZCR can indicate more noise
# or percussive sounds, while a lower ZCR can indicate more tonal sounds.
# High ZCR = more likely to be AI, low ZCR = more likely to be human.
# Sectional ZCR comparisons help detect unnatural consistency across a track.


# ── Spectral Centroid ─────────────────────────────────────────────────────────
# The "center of mass" of the spectrum — a higher centroid means more energy
# in high frequencies, indicating a brighter sound.
# AI music might have different centroid patterns than human music depending
# on how it was generated. Comparing centroid at different sections of the
# track (first 2s, mid-song, end-song) can reveal whether the tonal brightness
# shifts naturally over time (human) or stays unnaturally consistent (AI).


# ── Spectral Bandwidth ────────────────────────────────────────────────────────
# How spread out the frequencies are — a wider bandwidth means more frequencies
# are present, indicating a richer sound.
# Low bandwidth = higher chance of being AI.
# High bandwidth = higher chance of being human.
# AI-generated music may have a narrower, more controlled frequency spread.


# ── Spectral Rolloff ──────────────────────────────────────────────────────────
# How abrupt the drop-off is in the spectrum — can indicate noisiness or
# harshness. Higher rolloff means more energy in high frequencies, which can
# be a sign of noise or harshness.
# AI music might have different rolloff patterns than human music.


# ── Spectral Flatness ─────────────────────────────────────────────────────────
# How "tonal" vs "noisy" the sound is.
# A flatness close to 0 means the sound is more tonal (like a pure note — AI likely).
# A flatness close to 1 means the sound is more noise-like (like static — human likely).
# AI-generated music tends to have a more controlled, tonal character.


# ── Spectral Flux (Onset Strength) ───────────────────────────────────────────
# How much the spectrum changes over time — a higher flux indicates more
# dynamic changes in the music, while a lower flux indicates a more static sound.
# AI music might have different flux patterns than human music.
# Comparing flux in the first 2 seconds to the overall track can reveal
# whether a track starts with an unusually "perfect" or consistent onset
# pattern, which is more typical of AI-generated music.


# ── Beat Tracking / Tempo ─────────────────────────────────────────────────────
# Estimated BPM of the track. AI music tends to have a very consistent,
# locked tempo with minimal deviation, while human music naturally drifts
# slightly. This is one of the strongest single indicators.


# ── Inter-Beat Interval (IBI) Variance ───────────────────────────────────────
# How consistent the timing of beats is — human music often has some natural
# variability in beat timing (micro-timing), while AI music tends to have very
# consistent, metronomic beat intervals.
# A higher IBI variance = more human-like timing.
# A very low IBI variance = likely AI-generated music.


# ── Stereo Correlation ────────────────────────────────────────────────────────
# How correlated the left and right channels are — a very high correlation
# (close to 1) can indicate a mono or very narrow stereo image, which is more
# common in AI-generated music. Human music often has a wider stereo image
# with less correlation between channels due to natural recording techniques,
# microphone placement, room acoustics, and mixing decisions.


# ── Harmonic-Percussive Separation (HPSS) ────────────────────────────────────
# Separates the signal into harmonic (tonal/melodic) and percussive (drum/
# transient) components. Used to compute several derived features.


# ── Harmonic-Percussive Beat Offset ──────────────────────────────────────────
# How "offset" the timing of the harmonic and percussive elements are.
# If they are perfectly aligned (offset = 0), it can indicate "hyper-locked"
# timing, which is often a sign of AI-generated music.
# Human music usually has some natural offset between harmonic and percussive
# elements due to the physical nature of playing instruments.


# ── MFCCs (Mel-Frequency Cepstral Coefficients) ──────────────────────────────
# A compact representation of the short-term power spectrum of sound,
# commonly used in audio classification. MFCCs capture timbral texture.
# AI-generated music may have different MFCC distributions than human music,
# particularly in terms of how the timbral texture evolves over time.


# ── Delta MFCCs (Contrasted MFCCs) ───────────────────────────────────────────
# The rate of change of MFCCs over time — captures how quickly the timbral
# texture is changing. A very low delta MFCC can indicate a static or
# "frozen" timbre, which may be more characteristic of AI music.


# ── Spectral Contrast ─────────────────────────────────────────────────────────
# The difference in amplitude between spectral peaks and valleys across
# frequency sub-bands. High contrast = clear tonal peaks against a quieter
# background. AI music may show unnaturally high or consistent spectral contrast.


# ── Chroma Features ───────────────────────────────────────────────────────────
# Represent the energy distribution across the 12 pitch classes (C, C#, D, etc.)
# regardless of octave. Captures harmonic and tonal content.
# AI music may have different chroma patterns — potentially more "perfect"
# chord voicings or less tonal variation than human-performed music.


# ── Tempo Stability ───────────────────────────────────────────────────────────
# The variance of the tempogram — how much the tempo fluctuates across the
# track. A very low variance = very consistent tempo, which can indicate
# AI-generated music. Human music typically has some natural tempo variation,
# even in electronic genres.


# ── Rhythm Complexity ─────────────────────────────────────────────────────────
# A combined measure derived from the variance of normalised RMS and ZCR.
# Captures how complex and varied the rhythmic structure of the track is.
# Very low rhythm complexity can indicate AI-generated music with a simple,
# repetitive structure.


# ── Chroma Variance ───────────────────────────────────────────────────────────
# How much the harmonic/tonal content varies over the track. Low variance
# could indicate a repetitive harmonic structure, potentially AI-generated.


# ── Contrast Variance ─────────────────────────────────────────────────────────
# How much the spectral contrast changes over time. Low variance suggests
# a consistent spectral texture — possibly AI-generated.


# ── Dynamic Range ─────────────────────────────────────────────────────────────
# The difference between the loudest and quietest parts of the track.
# AI music can be heavily compressed (low dynamic range) or have an
# unnaturally consistent level throughout. Human music tends to have more
# natural dynamic variation.


# ── Rhythm Entropy ────────────────────────────────────────────────────────────
# A measure of randomness or unpredictability in the rhythmic energy
# distribution. Higher entropy = more varied, human-like rhythm.
# Lower entropy = more predictable, potentially AI-generated rhythm.


# ── Skewness & Kurtosis ───────────────────────────────────────────────────────
# Measures of the "shape" of the distribution of features like RMS.
# A high kurtosis indicates more extreme values (e.g. very loud peaks).
# A high skewness indicates asymmetry in the distribution (e.g. more quiet
# sections than loud sections).
# AI music might have different kurtosis and skewness patterns depending on
# how it was generated — often more symmetric and lower-kurtosis.


# ── Phase Incoherence ─────────────────────────────────────────────────────────
# The mean absolute second derivative of the STFT phase — a classic forensic
# measure of "phase jitter" or incoherence in the signal.
# Very low phase incoherence can indicate an unnaturally "perfect" or
# synthesized signal, potentially AI-generated.
# Human recordings naturally accumulate phase noise from microphones, room
# acoustics, and analog signal chains.


# ── Spectral Peakiness ────────────────────────────────────────────────────────
# The ratio of the maximum spectral centroid to the mean — how "spiky" the
# frequency content is over time. A very high peakiness can indicate
# unnatural frequency transients.


# ── Onset Jitter ─────────────────────────────────────────────────────────────
# The average distance between actual note onsets and the nearest theoretical
# beat position.
# Human: High jitter (avg > 0.015s) — natural micro-timing variation.
# AI:    Low jitter (avg < 0.005s) — robotic precision, notes land exactly
#        on the beat grid.
# This is one of the most reliable single features for AI detection.


# ── Pitch Stability ───────────────────────────────────────────────────────────
# The standard deviation of the fundamental frequency (f0) across voiced frames,
# extracted using pyin.
# AI-generated music might have more stable pitch (less variance) compared to
# human music, which has natural pitch fluctuations from vibrato, breath
# pressure, and performance expression.
# Human: High variance (> 1.5 Hz).
# AI:    Low variance (< 0.5 Hz).


# ── Fine Pitch Fluctuations ───────────────────────────────────────────────────
# The standard deviation of the pitch tracking matrix (piptrack) — captures
# subtle, frame-level pitch variation across all frequencies.
# AI-generated music may show less fine-grained pitch variation than human
# recordings, which contain natural micro-inflections.


# ── Breath Gaps ───────────────────────────────────────────────────────────────
# Detected using HPSS to isolate the harmonic (vocal/melodic) component, then
# splitting on silence.
# "Impossible breaths" (< 25ms gaps) are a strong AI indicator — no human
# singer can produce a breath gap that short. They typically arise from digital
# splicing or AI synthesis artifacts.
# "Unnatural phrases" (> 20s without a breath) are also suspicious — no human
# vocalist can sustain a phrase that long without breathing.


# ── RMS and Flux in First Few Seconds ────────────────────────────────────────
# AI music might have a more "perfect" or "consistent" start, while human music
# might have a more gradual build-up or some variability in the beginning.
# Comparing the RMS and flux in the first few seconds to the overall track
# can reveal differences in how AI and human music typically begin.


# ── Frequency Threshold Counts ───────────────────────────────────────────────
# The number of spectral frames where the centroid exceeds 10kHz, 5kHz, and 1kHz.
# AI music may have an unnaturally high or consistent number of high-frequency
# frames, or conversely an unusually low number, depending on the generation
# method used.


# ── Cross-Correlations ────────────────────────────────────────────────────────
# Centroid / Rolloff cross-correlation: how closely the brightness and
# roll-off frequency track each other over time. A very high correlation
# could indicate a uniform, AI-generated spectral envelope.
#
# Flatness / Contrast cross-correlation: how closely the tonality and
# spectral contrast track each other. Unusual cross-correlations between
# these features may indicate AI synthesis artifacts.