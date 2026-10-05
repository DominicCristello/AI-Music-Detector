import librosa
import numpy as np
import os
import scipy.stats


def measure_pitch_stability(y, sr):
    # Downsample to 11025 Hz for pyin — pitch tracking doesn't need 44.1k
    if sr > 11025:
        y = librosa.resample(y, orig_sr=sr, target_sr=11025)
        sr = 11025
    f0, _, _ = librosa.pyin(y, sr=sr,
                             fmin=librosa.note_to_hz('C2'),
                             fmax=librosa.note_to_hz('C7'),
                             frame_length=2048)
    f0_clean = f0[~np.isnan(f0)]
    return float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0


def measure_onset_jitter(y, sr):
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times   = librosa.frames_to_time(beat_frames, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)
    jitters = [np.abs(onset - beat_times[np.argmin(np.abs(beat_times - onset))])
               for onset in onset_times]
    return float(np.mean(jitters)) if jitters else 0.0


def count_breath_gaps(y, sr):
    y_harmonic, _ = librosa.effects.hpss(y)
    intervals = librosa.effects.split(y_harmonic, top_db=45)
    phrase_lengths    = [(e - s) / sr for s, e in intervals]
    unnatural_phrases = [p for p in phrase_lengths if p > 20.0]
    gaps = [(intervals[i+1][0] - intervals[i][1]) / sr
            for i in range(len(intervals) - 1)]
    impossible_breaths = [g for g in gaps if 0.0001 < g < 0.025]
    return {"unnatural_phrases": len(unnatural_phrases),
            "impossible_breaths": len(impossible_breaths)}


def _p(cb, pct, msg):
    if cb is not None:
        cb(pct, msg)


def compute_info(y, sr, filename, on_progress=None):
    cb = on_progress
    duration = y.shape[-1] / sr
    filename = os.path.basename(filename) if filename else "N/A"

    if duration < 60:
        return ("Track is too short for reliable analysis "
                "(less than 1 minute). Please provide a longer track.")
    if duration > 600:
        return ("Track is too long for reliable analysis "
                "(more than 10 minutes). Please provide a shorter track.")

    _p(cb, 0.02, "Converting stereo to mono...")
    y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
    y_mono = np.asarray(y_mono, dtype=np.float32)
    mid_s  = int(len(y_mono) / 2)

    _p(cb, 0.04, "Calculating RMS energy...")
    rms = librosa.feature.rms(y=y_mono)[0]

    _p(cb, 0.06, "Calculating RMS - first 5 seconds...")
    rms_first_5s = librosa.feature.rms(y=y_mono[:5*sr])[0]

    _p(cb, 0.08, "Calculating RMS - first 30 seconds...")
    rms_first_30s = librosa.feature.rms(y=y_mono[:30*sr])[0]

    _p(cb, 0.09, "Calculating RMS - middle 20 seconds...")
    rms_mid_20s = librosa.feature.rms(y=y_mono[mid_s-10*sr:mid_s+10*sr])[0]

    _p(cb, 0.10, "Calculating RMS - last 20 seconds...")
    rms_last_20s = librosa.feature.rms(y=y_mono[-20*sr:])[0]

    _p(cb, 0.11, "Calculating zero-crossing rate...")
    zcr = librosa.feature.zero_crossing_rate(y_mono)[0]

    _p(cb, 0.12, "Calculating ZCR - first 5 seconds...")
    zcr_first_5s = librosa.feature.zero_crossing_rate(y_mono[:5*sr])[0]

    _p(cb, 0.13, "Calculating ZCR - middle 20 seconds...")
    zcr_mid_20s = librosa.feature.zero_crossing_rate(y=y_mono[mid_s-10*sr:mid_s+10*sr])[0]

    _p(cb, 0.14, "Calculating ZCR - first 30 seconds...")
    zcr_first_30s = librosa.feature.zero_crossing_rate(y=y_mono[:30*sr])[0]

    _p(cb, 0.15, "Calculating ZCR - middle 30 seconds...")
    zcr_mid_30s = librosa.feature.zero_crossing_rate(y=y_mono[mid_s-15*sr:mid_s+15*sr])[0]

    _p(cb, 0.16, "Calculating ZCR - last 20 seconds...")
    zcr_last_20s = librosa.feature.zero_crossing_rate(y=y_mono[-20*sr:])[0]

    _p(cb, 0.18, "Calculating spectral centroid...")
    centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)[0]

    _p(cb, 0.20, "Calculating spectral bandwidth...")
    bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)[0]

    _p(cb, 0.22, "Calculating spectral rolloff...")
    rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)[0]

    _p(cb, 0.24, "Calculating spectral flatness...")
    flatness            = librosa.feature.spectral_flatness(y=y_mono)[0]
    flatness_diffs      = np.diff(flatness)
    abs_flatness_change = np.abs(flatness_diffs)

    _p(cb, 0.26, "Calculating onset flux...")
    flux = librosa.onset.onset_strength(y=y_mono, sr=sr)

    _p(cb, 0.27, "Calculating onset flux - first 2 seconds...")
    flux_first_2s = librosa.onset.onset_strength(y=y_mono[:2*sr], sr=sr)

    _p(cb, 0.29, "Tracking beats and estimating tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    tempo      = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    _p(cb, 0.33, "Computing inter-beat interval variance...")
    ibi_variance = float(np.var(np.diff(beat_times))) if len(beat_times) > 1 else 0.0

    _p(cb, 0.35, "Analysing stereo field correlation...")
    if y.ndim == 2 and y.shape[0] >= 2:
        left  = np.asarray(y[0], dtype=np.float32)
        right = np.asarray(y[1], dtype=np.float32)
        mask  = ~np.isnan(left) & ~np.isnan(right)
        left, right = left[mask], right[mask]
        if len(left) > 1:
            correlation = float(np.corrcoef(left, right)[0, 1])
            if np.isnan(correlation): correlation = 0.0
        else:
            correlation = 0.0
    else:
        correlation = 1.0

    _p(cb, 0.37, "Separating harmonic and percussive components...")
    y_harmonic, y_percussive = librosa.effects.hpss(y_mono)

    _p(cb, 0.43, "Tracking harmonic beats...")
    tempo_h, beats_h = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    tempo_h = float(np.atleast_1d(tempo_h)[0])

    _p(cb, 0.47, "Tracking percussive beats...")
    tempo_p, beats_p = librosa.beat.beat_track(y=y_percussive, sr=sr)
    tempo_p = float(np.atleast_1d(tempo_p)[0])

    _p(cb, 0.50, "Computing harmonic-percussive beat offset...")
    min_len = min(len(beats_h), len(beats_p))
    if min_len > 0:
        bt_h    = librosa.frames_to_time(beats_h[:min_len], sr=sr)
        bt_p    = librosa.frames_to_time(beats_p[:min_len], sr=sr)
        offsets = np.abs(bt_h - bt_p)
        mean_offset = float(np.mean(offsets))
    else:
        offsets = np.array([])
        mean_offset = 0.0

    _p(cb, 0.52, "Calculating MFCCs...")
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)

    _p(cb, 0.55, "Calculating delta MFCCs...")
    contrasted_mfcc = librosa.feature.delta(mfcc)

    _p(cb, 0.57, "Calculating spectral contrast...")
    contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)

    _p(cb, 0.59, "Calculating chroma features...")
    chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)

    _p(cb, 0.61, "Computing tempogram and tempo stability...")
    tempogram       = librosa.feature.tempogram(y=y_mono, sr=sr)
    tempo_stability = np.var(tempogram)
    tempogram_std   = np.std(tempogram)

    _p(cb, 0.65, "Calculating rhythm complexity...")
    rms_vals = librosa.feature.rms(y=y_mono)[0]
    zcr_vals = librosa.feature.zero_crossing_rate(y_mono)[0]
    rythym_complexity = np.var(rms_vals) * np.var(zcr_vals)

    _p(cb, 0.67, "Calculating chroma and contrast variance...")
    chroma_variance   = np.var(chroma)
    contrast_variance = np.var(contrast)

    _p(cb, 0.68, "Calculating dynamic range...")
    dynamic_range = float(np.max(rms_vals) - np.min(rms_vals))

    _p(cb, 0.69, "Calculating rhythm entropy...")
    rms_prob      = rms_vals / (np.sum(rms_vals) + 1e-10)
    rythym_entropy = -np.sum(rms_prob * np.log(rms_prob + 1e-10))

    _p(cb, 0.70, "Calculating amplitude skewness...")
    skewness = scipy.stats.skew(rms_vals)

    _p(cb, 0.71, "Calculating amplitude kurtosis...")
    kurtosis = scipy.stats.kurtosis(rms_vals)

    _p(cb, 0.72, "Computing STFT...")
    D = librosa.stft(y_mono)

    _p(cb, 0.74, "Computing phase acceleration...")
    phase       = np.angle(D)
    phase_accel = np.diff(phase, axis=1, n=2)

    _p(cb, 0.75, "Calculating phase incoherence...")
    phase_incoherence = float(np.mean(np.abs(phase_accel)))

    _p(cb, 0.76, "Calculating spectral peakiness...")
    spectral_peakiness = float(np.max(centroid) / (np.mean(centroid) + 1e-10))

    _p(cb, 0.77, "Measuring onset jitter...")
    measure_onset_jitter_value = measure_onset_jitter(y_mono, sr)

    _p(cb, 0.81, "Measuring pitch stability (pyin)...")
    measure_pitch_stability_value = measure_pitch_stability(y_mono, sr)

    _p(cb, 0.86, "Detecting breath gaps...")
    count_breath_gaps_value = count_breath_gaps(y_mono, sr)

    _p(cb, 0.91, "Calculating fine pitch fluctuations...")
    pitch_fluctuations = librosa.core.piptrack(y=y_mono, sr=sr)[0]
    pitch_fluct_std    = float(np.std(pitch_fluctuations) if pitch_fluctuations.size > 0 else 0.0)

    if np.max(np.abs(y)) < 1e-6:
        return "Track is effectively silent. Please provide a valid audio track."

    _p(cb, 0.94, "Compiling sectional statistics...")
    mid_rolloff  = len(rolloff)  // 2
    end_rolloff  = int(len(rolloff)  * 0.75)
    mid_flatness = len(flatness) // 2
    end_flatness = int(len(flatness) * (3/4))
    mid_chroma   = chroma.shape[1] // 2
    window       = 120

    _p(cb, 0.96, "Compiling results...")
    lines = [
        "File info:\n",
        f"  File Name: {filename}",
        f"  Sample Rate: {sr} Hz",
        f"  Duration: {duration:.2f} s",
        "Summary features",
        f"  Tempo (estimate): {tempo:.2f} BPM",
        f"  RMS_Mean: mean={rms.mean():.6f}",
        f"  RMS_Std: std={rms.std():.6f}",
        f"  RMS_Mean in first 5s: mean={rms_first_5s.mean():.6f}",
        f"  RMS_Std in first 5s: std={rms_first_5s.std():.6f}",
        f"  RMS_Mean in first 30 seconds: mean={rms_first_30s.mean():.6f}",
        f"  RMS_Std in first 30 seconds: std={rms_first_30s.std():.6f}",
        f"  RMS_Mean in middle 20 seconds: mean={rms_mid_20s.mean():.6f}",
        f"  RMS_Std in middle 20 seconds: std={rms_mid_20s.std():.6f}",
        f"  RMS_Mean in last 20 seconds: mean={rms_last_20s.mean():.6f}",
        f"  RMS_Std in last 20 seconds: std={rms_last_20s.std():.6f}",
        f"  ZCR_Mean: mean={zcr.mean():.6f}",
        f"  ZCR_Std: std={zcr.std():.6f}",
        f"  ZCR_Mean in first 5s: mean={zcr_first_5s.mean():.6f}",
        f"  ZCR_Std in first 5s: std={zcr_first_5s.std():.6f}",
        f"  ZCR_Mean in middle 20 seconds: mean={zcr_mid_20s.mean():.6f}",
        f"  ZCR_Std in middle 20 seconds: std={zcr_mid_20s.std():.6f}",
        f"  ZCR_Mean in first 30 seconds: mean={zcr_first_30s.mean():.6f}",
        f"  ZCR_Std in first 30 seconds: std={zcr_first_30s.std():.6f}",
        f"  ZCR_Mean in middle 30 seconds: mean={zcr_mid_30s.mean():.6f}",
        f"  ZCR_Std in middle 30 seconds: std={zcr_mid_30s.std():.6f}",
        f"  ZCR_Mean in last 20 seconds: mean={zcr_last_20s.mean():.6f}",
        f"  ZCR_Std in last 20 seconds: std={zcr_last_20s.std():.6f}",
        f"  Mean_Spectral_Centroid: mean={centroid.mean():.2f} Hz",
        f"  Spectral_Centroid Std: std={centroid.std():.2f} Hz",
        f"  Mean_Spectral_Centroid in First 2s: mean={centroid[:int(2*sr/512)].mean():.2f} Hz",
        f"  Spectral_Centroid_Std in First 2s: std={centroid[:int(2*sr/512)].std():.2f} Hz",
        f"  Mean_Spectral_Centroid_Mid-Song: mean={np.mean(centroid[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Spectral_Centroid_Std Mid-Song: std={np.std(centroid[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Centroid End-Song: mean={np.mean(centroid[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Spectral_Centroid_Std End-Song: std={np.std(centroid[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Bandwidth: mean={bandwidth.mean():.2f} Hz",
        f"  Spectral_Bandwidth_Std: std={bandwidth.std():.2f} Hz",
        f"  Mean_Spectral_Bandwidth in First 2s: mean={bandwidth[:int(2*sr/512)].mean():.2f} Hz",
        f"  Spectral_Bandwidth_Std in First 2s: std={bandwidth[:int(2*sr/512)].std():.2f} Hz",
        f"  Mean_Spectral_Bandwidth Mid-Song: mean={np.mean(bandwidth[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Spectral_Bandwidth_Std Mid-Song: std={np.std(bandwidth[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Bandwidth End-Song: mean={np.mean(bandwidth[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Spectral_Bandwidth_Std End-Song: std={np.std(bandwidth[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Flatness: mean={flatness.mean():.6f}",
        f"  Spectral_Flatness_Std: std={flatness.std():.6f}",
        f"  Mean_Spectral_Flatness in First 2s: mean={flatness[:int(2*sr/512)].mean():.6f}",
        f"  Spectral_Flatness_Std in First 2s: std={flatness[:int(2*sr/512)].std():.6f}",
        f"  Mean_Spectral_Flatness Mid-Song: mean={np.mean(flatness[mid_flatness-window:mid_flatness+window]):.6f}",
        f"  Spectral_Flatness_Std Mid-Song: std={np.std(flatness[mid_flatness-window:mid_flatness+window]):.6f}",
        f"  Mean_Spectral_Flatness End-Song: mean={np.mean(flatness[end_flatness-window:end_flatness+window]):.6f}",
        f"  Spectral_Flatness_Std End-Song: std={np.std(flatness[end_flatness-window:end_flatness+window]):.6f}",
        f"  Mean_Spectral_Flatness_Change: mean={np.mean(abs_flatness_change):.6f}",
        f"  Spectral_Flatness_Change_Volatility: std={np.std(flatness_diffs):.6f}",
        f"  Spectral Entroid Variance: {np.var(centroid):.6f}",
        f"  Mean_Spectral_Flux: mean={flux.mean():.6f}",
        f"  Spectral_Flux_Std: std={flux.std():.6f}",
        f"  Mean_Spectral_Flux in First 2s: mean={flux_first_2s.mean():.6f}",
        f"  Spectral_Flux_Std in First 2s: std={flux_first_2s.std():.6f}",
        f"  Mean_Spectral_Flux Mid-Song: mean={np.mean(flux[int(len(flux)/2)-window:int(len(flux)/2)+window]):.6f}",
        f"  Spectral_Flux_Std Mid-Song: std={np.std(flux[int(len(flux)/2)-window:int(len(flux)/2)+window]):.6f}",
        f"  Mean_Spectral_Flux End-Song: mean={np.mean(flux[-window:]):.6f}",
        f"  Spectral_Flux_Std End-Song: std={np.std(flux[-window:]):.6f}",
        f"  Mean_Average_Bandwidth: mean={bandwidth.mean():.2f} Hz",
        f"  Average_Bandwidth_Std: std={bandwidth.std():.2f} Hz",
        f"  Mean_Average_Bandwidth in First 2s: mean={bandwidth[:int(2*sr/512)].mean():.2f} Hz",
        f"  Average_Bandwidth_Std in First 2s: std={bandwidth[:int(2*sr/512)].std():.2f} Hz",
        f"  Mean_Average_Bandwidth Mid-Song: mean={np.mean(bandwidth[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Average_Bandwidth_Std Mid-Song: std={np.std(bandwidth[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Mean_Average_Bandwidth End-Song: mean={np.mean(bandwidth[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Average_Bandwidth_Std End-Song: std={np.std(bandwidth[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Rolloff: mean={rolloff.mean():.2f} Hz",
        f"  Spectral_Rolloff_Std: std={rolloff.std():.2f} Hz",
        f"  Mean_Spectral_Rolloff in First 2s: mean={rolloff[:int(2*sr/512)].mean():.2f} Hz",
        f"  Spectral_Rolloff_Std in First 2s: std={rolloff[:int(2*sr/512)].std():.2f} Hz",
        f"  Mean_Spectral_Rolloff Mid-Song: mean={np.mean(rolloff[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Spectral_Rolloff_Std Mid-Song: std={np.std(rolloff[mid_rolloff-window:mid_rolloff+window]):.2f} Hz",
        f"  Mean_Spectral_Rolloff End-Song: mean={np.mean(rolloff[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Spectral_Rolloff_Std End-Song: std={np.std(rolloff[end_rolloff-window:end_rolloff+window]):.2f} Hz",
        f"  Mean_Average_Spectral_Flatness: mean={flatness.mean():.6f}",
        f"  Spectral_Flatness_Std: std={flatness.std():.6f}",
        f"  Mean Absolute Flatness Change: mean={np.mean(abs_flatness_change):.6f}",
        f"  Volatility of Flatness: std={np.std(flatness_diffs):.6f}",
        f"  IBI Variance: {ibi_variance:.6f}",
        f"  Left-Right Correlation: {'N/A (mono track)' if y.ndim == 1 else f'{correlation:.6f}'}",
        f"  Mean_Harmonic-Percussive_Offset: mean={mean_offset:.6f} s, std={np.std(offsets) if len(offsets) > 0 else 0.0:.6f} s",
        f"  Harmonic Tempo: {tempo_h:.2f} BPM",
        f"  Percussive Tempo: {tempo_p:.2f} BPM",
        f"  Amplitude of Harmonic Component: {np.mean(np.abs(y_harmonic)):.6f}",
        f"  Mono vs Stereo: {'Mono' if y.ndim == 1 else 'Stereo'}",
        f"  Mean_MFCC: mean={mfcc.mean():.6f}",
        f"  MFCC_Std: std={mfcc.std():.6f}",
        f"  Contrasted_Mean_MFCC: mean={contrasted_mfcc.mean():.6f}",
        f"  Contrasted_MFCC_Std: std={contrasted_mfcc.std():.6f}",
        f"  Mean_Spectral_Contrast: mean={contrast.mean():.6f}",
        f"  Spectral_Contrast_Std: std={contrast.std():.6f}",
        f"  Mean_Chroma: mean={chroma.mean():.6f}",
        f"  Chroma_Std: std={chroma.std():.6f}",
        f"  Mean_Chroma in First 2s: mean={chroma[:,:int(2*sr/512)].mean():.6f}",
        f"  Chroma_Std in First 2s: std={chroma[:,:int(2*sr/512)].std():.6f}",
        f"  Mean_Chroma Mid-Song: mean={np.mean(chroma[:,mid_chroma-window:mid_chroma+window]):.6f}",
        f"  Chroma_Std Mid-Song: std={np.std(chroma[:,mid_chroma-window:mid_chroma+window]):.6f}",
        f"  Mean_Chroma End-Song: mean={np.mean(chroma[:,-window:]):.6f}",
        f"  Chroma_Std End-Song: std={np.std(chroma[:,-window:]):.6f}",
        f"  Mean_Contrast in First 2s: mean={contrast[:,:int(2*sr/512)].mean():.6f}",
        f"  Contrast_Std in First 2s: std={contrast[:,:int(2*sr/512)].std():.6f}",
        f"  Mean_Contrast Mid-Song: mean={np.mean(contrast[:,mid_rolloff-window:mid_rolloff+window]):.6f}",
        f"  Contrast_Std Mid-Song: std={np.std(contrast[:,mid_rolloff-window:mid_rolloff+window]):.6f}",
        f"  Mean_Contrast End-Song: mean={np.mean(contrast[:,-window:]):.6f}",
        f"  Contrast_Std End-Song: std={np.std(contrast[:,-window:]):.6f}",
        f"  Tempo Stability: {tempo_stability:.6f}",
        f"  Standard Deviation of Tempo: {tempogram_std:.6f}",
        f"  Rythym Complexity: {rythym_complexity:.6f}",
        f"  Chroma Variance: {chroma_variance:.6f}",
        f"  Contrast Variance: {contrast_variance:.6f}",
        f"  Dynamic Range: {dynamic_range:.6f}",
        f"  Rythym Entropy: {rythym_entropy:.6f}",
        f"  Skewness: {skewness:.6f}",
        f"  Kurtosis: {kurtosis:.6f}",
        f"  Spectral Peakiness: {spectral_peakiness:.6f}",
        f"  Phase Incoherence: {phase_incoherence:.6f}",
        f"  Onset Jitter: {measure_onset_jitter_value:.6f} s",
        f"  Pitch Stability: {measure_pitch_stability_value:.6f} Hz",
        f"  Fine-Pitch Fluctuations: std={pitch_fluct_std:.6f} Hz",
        f"  Breath Gaps: {count_breath_gaps_value}",
        f"  Number of Times the Frequency Exceeds 10kHz: {np.sum(centroid > 10000)}",
        f"  Number of Times the Frequency Exceeds 5kHz: {np.sum(centroid > 5000)}",
        f"  Number of Times the Frequency Exceeds 1kHz: {np.sum(centroid > 1000)}",
        f"  Cross-Correlation (Centroid/Rolloff): {np.corrcoef(centroid, rolloff)[0, 1]:.6f}",
        f"  Cross-Correlation (Flatness/Contrast): {np.corrcoef(flatness, np.mean(contrast, axis=0))[0, 1]:.6f}",
    ]
    return "\n".join(lines)

#SPECTRAL ROLLOFF: how abrupt the dropoff is in the spectrum, can indicate noisiness or harshness -
# higher rolloff means more energy in high frequencies, which can be a sign of noise or harshness.
# AI music might have different rolloff patterns than human music

#SPECTRAL FLATNESS: how "tonal" vs "noisy" the sound is - a flatness close to 0 means the sound is more tonal
# (like a pure note - AI likely), while a flatness close to 1 means the sound is more noise-like (like static -
# human like).

#INTER-BEAT INTERVAL (IBI) VARIANCE: how consistent the timing of beats is - human music often has some natural
# variability in beat timing, while AI music might have very consistent beat intervals. A higher IBI variance could
# indicate more human-like timing, while a very low IBI variance might suggest AI-generated music.

# BANDWIDTH: how spread out the frequencies are - a wider bandwidth means more frequencies are present, which can
# indicate a richer sound. Low bandwidth = higher chance of being AI, high bandwidth = high chance of being human

# CENTROID: the "center of mass" of the spectrum - a higher centroid means more energy in high frequencies, which can
# indicate a brighter sound. AI music might have different centroid patterns than human music, depending on how it was generated.

# ZCR (Zero-Crossing Rate): how often the signal crosses zero - a higher ZCR can indicate more noise or percussive sounds,
# while a lower ZCR can indicate more tonal sounds. High ZCR = more likely to be AI, low ZCR = more likely to be human

#RMS (Root Mean Square): a measure of the overall energy of the signal - a higher RMS can indicate a louder sound, while a lower
# RMS can indicate a quieter sound. AI music might have different RMS patterns than human music, depending on how it was generated.

# HARMONIC-PERCUSSIVE OFFSET: how "offset" the timing of the harmonic and percussive elements are - if they are perfectly aligned
# (offset = 0), it can indicate "hyper-locked" timing which is often a sign of AI-generated music. Human music usually has some natural
# offset between these elements.

# STEREO CORRELATION: how correlated the left and right channels are - a very high correlation (close to 1) can indicate a mono or very narrow stereo image, which might be more common in AI-generated music. Human music often has a wider stereo image with less correlation between channels.

#SPECTRAL FLUX: how much the spectrum changes over time - a higher flux can indicate more dynamic changes in the music, while a lower flux can indicate a more static sound. AI music might have different flux patterns than human music, depending on how it was generated.

#KURTOSIS AND SKEWNESS: measures of the "shape" of the distribution of features like RMS or ZCR - a high kurtosis can indicate more extreme values (e.g. very loud peaks), while a high skewness can indicate an asymmetry in the distribution (e.g. more quiet sections than loud sections). AI music might have different kurtosis and skewness patterns than human music, depending on how it was generated.

#RMS AND FLUX IN FIRST FEW SECONDS: AI music might have a more "perfect" or "consistent" start, while human music might have a more gradual build-up or some variability in the beginning. Comparing the RMS and flux in the first few seconds to the overall track could reveal differences in how AI and human music typically start.

#PITCH STABILITY: AI-generated music might have more stable pitch (less variance in the fundamental frequency) compared to human music, which can have more
# natural fluctuations in pitch.
