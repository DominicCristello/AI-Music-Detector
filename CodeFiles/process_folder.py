#RUN: python build_dataset.py for human tracks

#RUN: python ingest_sonics.py, then python generate_ai_tracks.py

import os
import csv
import librosa
import numpy as np
from glob import glob
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')  # Suppress librosa warnings for cleaner output
from computeinfo import compute_info
#this function will take a lot of computing power (such as calculating the MFCCs, chroma features, spectral contrast, etc.) so we
# will run it separately from the GUI and save the results to a CSV file. This way, we can build a large dataset of features for
# both AI and human music, which can then be used to train a machine learning model to classify new tracks as AI or human generated
# based on their audio features.



FIELD_MAP = {
    'filename': 'fileName',
    'label': 'artificiallyGenerated',
    'stereo_correlation': 'stereoCorrelation',
    'sample_rate': 'sampleRate[Hz]',
    'duration': 'duration[s]',
    'tempo_estimate': 'tempoEstimate[BPM]',
    'rms_mean_mean': 'rmsMean[amp]',
    'rms_std_std': 'rmsStd[amp]',
    'rms_mean_in_first_5s_mean': 'rmsMeanFirst5s[amp]',
    'rms_std_in_first_5s_std': 'rmsStdFirst5s[amp]',
    'rms_mean_in_first_30_seconds_mean': 'rmsMeanFirst30s[amp]',
    'rms_std_in_first_30_seconds_std': 'rmsStdFirst30s[amp]',
    'rms_mean_in_middle_20_seconds_mean': 'rmsMeanMid20s[amp]',
    'rms_std_in_middle_20_seconds_std': 'rmsStdMid20s[amp]',
    'rms_mean_in_last_20_seconds_mean': 'rmsMeanLast20s[amp]',
    'rms_std_in_last_20_seconds_std': 'rmsStdLast20s[amp]',
    'zcr_mean_mean': 'zcrMean',
    'zcr_std_std': 'zcrStd',
    'zcr_mean_in_first_5s_mean': 'zcrMeanFirst5s',
    'zcr_std_in_first_5s_std': 'zcrStdFirst5s',
    'zcr_mean_in_middle_20_seconds_mean': 'zcrMeanMid20s',
    'zcr_std_in_middle_20_seconds_std': 'zcrStdMid20s',
    'zcr_mean_in_first_30_seconds_mean': 'zcrMeanFirst30s',
    'zcr_std_in_first_30_seconds_std': 'zcrStdFirst30s',
    'zcr_mean_in_middle_30_seconds_mean': 'zcrMeanMid30s',
    'zcr_std_in_middle_30_seconds_std': 'zcrStdMid30s',
    'zcr_mean_in_last_20_seconds_mean': 'zcrMeanLast20s',
    'zcr_std_in_last_20_seconds_std': 'zcrStdLast20s',
    'mean_spectral_centroid_mean': 'spectralCentroidMean[Hz]',
    'spectral_centroid_std_std': 'spectralCentroidStd[Hz]',
    'mean_spectral_centroid_in_first_2s_mean': 'spectralCentroidMeanFirst2s[Hz]',
    'spectral_centroid_std_in_first_2s_std': 'spectralCentroidStdFirst2s[Hz]',
    'mean_spectral_centroid_mid-song_mean': 'spectralCentroidMeanMidSong[Hz]',
    'spectral_centroid_std_mid-song_std': 'spectralCentroidStdMidSong[Hz]',
    'mean_spectral_centroid_end-song_mean': 'spectralCentroidMeanEndSong[Hz]',
    'spectral_centroid_std_end-song_std': 'spectralCentroidStdEndSong[Hz]',
    'mean_spectral_bandwidth_mean': 'spectralBandwidthMean[Hz]',
    'spectral_bandwidth_std_std': 'spectralBandwidthStd[Hz]',
    'mean_spectral_bandwidth_in_first_2s_mean': 'spectralBandwidthMeanFirst2s[Hz]',
    'spectral_bandwidth_std_in_first_2s_std': 'spectralBandwidthStdFirst2s[Hz]',
    'mean_spectral_bandwidth_mid-song_mean': 'spectralBandwidthMeanMidSong[Hz]',
    'spectral_bandwidth_std_mid-song_std': 'spectralBandwidthStdMidSong[Hz]',
    'mean_spectral_bandwidth_end-song_mean': 'spectralBandwidthMeanEndSong[Hz]',
    'spectral_bandwidth_std_end-song_std': 'spectralBandwidthStdEndSong[Hz]',
    'mean_spectral_flatness_mean': 'spectralFlatnessMean',
    'spectral_flatness_std_std': 'spectralFlatnessStd',
    'mean_spectral_flatness_in_first_2s_mean': 'spectralFlatnessMeanFirst2s',
    'spectral_flatness_std_in_first_2s_std': 'spectralFlatnessStdFirst2s',
    'mean_spectral_flatness_mid-song_mean': 'spectralFlatnessMeanMidSong',
    'spectral_flatness_std_mid-song_std': 'spectralFlatnessStdMidSong',
    'mean_spectral_flatness_end-song_mean': 'spectralFlatnessMeanEndSong',
    'spectral_flatness_std_end-song_std': 'spectralFlatnessStdEndSong',
    'mean_spectral_flatness_change_mean': 'spectralFlatnessChangeMean',
    'spectral_flatness_change_volatility_std': 'spectralFlatnessChangeVolatilityStd',
    'spectral_entroid_variance': 'spectralCentroidVariance[Hz2]',
    'mean_spectral_flux_mean': 'spectralFluxMean',
    'spectral_flux_std_std': 'spectralFluxStd',
    'mean_spectral_flux_in_first_2s_mean': 'spectralFluxMeanFirst2s',
    'spectral_flux_std_in_first_2s_std': 'spectralFluxStdFirst2s',
    'mean_spectral_flux_mid-song_mean': 'spectralFluxMeanMidSong',
    'spectral_flux_std_mid-song_std': 'spectralFluxStdMidSong',
    'mean_spectral_flux_end-song_mean': 'spectralFluxMeanEndSong',
    'spectral_flux_std_end-song_std': 'spectralFluxStdEndSong',
    'mean_average_bandwidth_mean': 'avgBandwidthMean[Hz]',
    'average_bandwidth_std_std': 'avgBandwidthStd[Hz]',
    'mean_average_bandwidth_in_first_2s_mean': 'avgBandwidthMeanFirst2s[Hz]',
    'average_bandwidth_std_in_first_2s_std': 'avgBandwidthStdFirst2s[Hz]',
    'mean_average_bandwidth_mid-song_mean': 'avgBandwidthMeanMidSong[Hz]',
    'average_bandwidth_std_mid-song_std': 'avgBandwidthStdMidSong[Hz]',
    'mean_average_bandwidth_end-song_mean': 'avgBandwidthMeanEndSong[Hz]',
    'average_bandwidth_std_end-song_std': 'avgBandwidthStdEndSong[Hz]',
    'mean_spectral_rolloff_mean': 'spectralRolloffMean[Hz]',
    'spectral_rolloff_std_std': 'spectralRolloffStd[Hz]',
    'mean_spectral_rolloff_in_first_2s_mean': 'spectralRolloffMeanFirst2s[Hz]',
    'spectral_rolloff_std_in_first_2s_std': 'spectralRolloffStdFirst2s[Hz]',
    'mean_spectral_rolloff_mid-song_mean': 'spectralRolloffMeanMidSong[Hz]',
    'spectral_rolloff_std_mid-song_std': 'spectralRolloffStdMidSong[Hz]',
    'mean_spectral_rolloff_end-song_mean': 'spectralRolloffMeanEndSong[Hz]',
    'spectral_rolloff_std_end-song_std': 'spectralRolloffStdEndSong[Hz]',
    'mean_average_spectral_flatness_mean': 'avgSpectralFlatnessMean',
    'mean_absolute_flatness_change_mean': 'absFlatnessChangeMean',
    'volatility_of_flatness_std': 'flatnessVolatilityStd',
    'ibi_variance': 'ibiVariance[s2]',
    'left-right_correlation': 'leftRightCorrelation',
    'mean_harmonic-percussive_offset_mean': 'harmonicPercussiveOffsetMean[s]',
    'mean_harmonic-percussive_offset_std': 'harmonicPercussiveOffsetStd[s]',
    'harmonic_tempo': 'harmonicTempo[BPM]',
    'percussive_tempo': 'percussiveTempo[BPM]',
    'amplitude_of_harmonic_component': 'harmonicAmplitude[amp]',
    'mean_mfcc_mean': 'mfccMean',
    'mfcc_std_std': 'mfccStd',
    'contrasted_mean_mfcc_mean': 'mfccDeltaMean',
    'contrasted_mfcc_std_std': 'mfccDeltaStd',
    'mean_spectral_contrast_mean': 'spectralContrastMean[dB]',
    'spectral_contrast_std_std': 'spectralContrastStd[dB]',
    'mean_chroma_mean': 'chromaMean',
    'chroma_std_std': 'chromaStd',
    'mean_chroma_in_first_2s_mean': 'chromaMeanFirst2s',
    'chroma_std_in_first_2s_std': 'chromaStdFirst2s',
    'mean_chroma_mid-song_mean': 'chromaMeanMidSong',
    'chroma_std_mid-song_std': 'chromaStdMidSong',
    'mean_chroma_end-song_mean': 'chromaMeanEndSong',
    'chroma_std_end-song_std': 'chromaStdEndSong',
    'mean_contrast_in_first_2s_mean': 'spectralContrastMeanFirst2s[dB]',
    'contrast_std_in_first_2s_std': 'spectralContrastStdFirst2s[dB]',
    'mean_contrast_mid-song_mean': 'spectralContrastMeanMidSong[dB]',
    'contrast_std_mid-song_std': 'spectralContrastStdMidSong[dB]',
    'mean_contrast_end-song_mean': 'spectralContrastMeanEndSong[dB]',
    'contrast_std_end-song_std': 'spectralContrastStdEndSong[dB]',
    'tempo_stability': 'tempoStability',
    'standard_deviation_of_tempo': 'tempoStd[BPM]',
    'rythym_complexity': 'rhythmComplexity',
    'chroma_variance': 'chromaVariance',
    'contrast_variance': 'contrastVariance[dB2]',
    'dynamic_range': 'dynamicRange[amp]',
    'rythym_entropy': 'rhythmEntropy[nats]',
    'skewness': 'rmsSkewness',
    'kurtosis': 'rmsKurtosis',
    'spectral_peakiness': 'spectralPeakiness',
    'phase_incoherence': 'phaseIncoherence[rad]',
    'onset_jitter': 'onsetJitter[s]',
    'pitch_stability': 'pitchStability[Hz]',
    'fine-pitch_fluctuations_std': 'finePitchFluctuationsStd[Hz]',
    'number_of_times_the_frequency_exceeds_10khz': 'freqExceeds10kHz[count]',
    'number_of_times_the_frequency_exceeds_5khz': 'freqExceeds5kHz[count]',
    'number_of_times_the_frequency_exceeds_1khz': 'freqExceeds1kHz[count]',
    'cross-correlation_centroid/rolloff': 'crossCorrCentroidRolloff',
    'cross-correlation_flatness/contrast': 'crossCorrFlatnessContrast',
}








# Audio containers we accept. FLAC/OGG added so lossless masters and codec-augmented
# variants are picked up too. Defined once so the two globs below can't drift apart.
AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.ogg")


def find_audio_files(folder_path):
    """All audio files in folder_path matching any accepted extension."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files += glob(os.path.join(folder_path, ext))
    return files


def _process_single_file(filename):
    """
    Worker function: processes a single audio file and returns a dictionary of features.
    This runs in a separate process, so it can't write to shared CSV.
    Returns (success, info_dict) tuple.
    """
    try:
        # 1. Load the original file in STEREO at 44.1kHz (Full Fidelity)
        y_stereo, sr = librosa.load(filename, sr=44100, mono=False)
        info_dict = {}

       # 2. Calculate SPATIAL features (Stereo Width / Phase)
        # MP3s can load with channels of slightly different lengths — fix inhomogeneous array
        if y_stereo.ndim == 1:
            y_stereo = np.stack([y_stereo, y_stereo]).astype(np.float32)  # mono: duplicate channel
        elif y_stereo.dtype == object:
            min_len = min(len(ch) for ch in y_stereo)
            y_stereo = np.array([np.array(ch[:min_len], dtype=np.float32) for ch in y_stereo], dtype=np.float32)
        else:
            y_stereo = np.array(y_stereo, dtype=np.float32)
        if y_stereo.ndim == 1:
            y_stereo = np.stack([y_stereo, y_stereo])
        if y_stereo.shape[0] > 2:
            y_stereo = y_stereo[:2]

        # Stereo correlation
        left, right = y_stereo[0], y_stereo[1]
        mask = ~np.isnan(left) & ~np.isnan(right)
        left, right = left[mask], right[mask]
        if len(left) > 1:
            corr = np.corrcoef(left, right)[0, 1]
            info_dict['stereo_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        else:
            info_dict['stereo_correlation'] = 0.0

        duration = y_stereo.shape[-1] / sr
        if duration < 60:
            return (False, os.path.basename(filename), "Track is too short (< 1 minute)")
        if duration > 600:
            return (False, os.path.basename(filename), "Track is too long (> 10 minutes)")

        # 3. Get other 30+ features


        info = compute_info(y_stereo, sr, filename)
        split_info = info.splitlines()
        for line in split_info:
            if ": " in line:
                key, value = line.split(": ", 1)
                base_key = key.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")

                # Handle lines with sub-keys like "mean=X, std=Y" or "mean=X s, std=Y s"
                if "=" in value:
                    sub_parts = value.split(",")
                    for part in sub_parts:
                        if "=" not in part:
                            continue
                        sub_key, sub_val = part.split("=", 1)
                        final_key = f"{base_key}_{sub_key.strip().lower()}"
                        numeric_val = "".join(c for c in sub_val if c.isdigit() or c in ".-")
                        if numeric_val:
                            try:
                                info_dict[final_key] = float(numeric_val)
                            except ValueError:
                                pass

                # Handle single value lines like "Tempo: 120 BPM" or "IBI Variance: 0.001234"
                else:
                    numeric_val = "".join(c for c in value if c.isdigit() or c in ".-")
                    if numeric_val:
                        try:
                            info_dict[base_key] = float(numeric_val)
                        except ValueError:
                            pass

        # Add metadata and label
        info_dict['filename'] = os.path.basename(filename)
        info_dict['label'] = 0  # Use 1 for AI, 0 for Human

        return (True, info_dict, None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (False, os.path.basename(filename), str(e))

def process_folder(folder_path, output_csv, num_workers=4):

    # Fieldnames must match the KEYS in the dictionary returned by compute_info
    # Run once on the first file just to get fieldnames
    sample_file = find_audio_files(folder_path)

    if not sample_file:
        print(f"No audio files found in {folder_path}. Please check the path and try again.")
        return


    fieldnames = list(FIELD_MAP.values())

    # Get set of already-processed files for resume functionality
    processed = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row.get('fileName') or row.get('filename', ''))

    # FIX: removed sample_file[0] from processed so it gets processed normally like all other files

    # Get all files and filter out already-processed ones
    all_files = find_audio_files(folder_path)
    files_to_process = [f for f in all_files if os.path.basename(f) not in processed]

    total_files = len(all_files)
    already_processed = len(processed)

    print(f"Total files in folder: {total_files}")
    print(f"Already processed: {already_processed}")
    print(f"Files to process: {len(files_to_process)}")
    print(f"Using {num_workers} parallel workers")
    print("This may take a while, especially for longer tracks...")

    if not files_to_process:
        print("All files have already been processed!")
        return

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_file, f): f for f in files_to_process}

        buffer = []
        processed_count = 0
        # FIX: use as_completed so results are handled as they finish
        for future in as_completed(futures):
            filename = futures[future]
            try:
                success, data, error_msg = future.result()

                if success:
                    data = {FIELD_MAP.get(k, k): v for k, v in data.items()}
                    buffer.append(data)
                    processed_count += 1
                    print(f"Successfully audited: {os.path.basename(filename)}")

                    # Flush buffer every 10 successful items
                    if len(buffer) >= 1:
                        write_batch(buffer, output_csv, fieldnames)
                        buffer = []
                else:
                    print(f"Skipping {data}: {error_msg}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Flush remaining buffer at end
        if buffer:
            write_batch(buffer, output_csv, fieldnames)

    print(f"\nProcessing complete! Successfully processed {processed_count} files.")


# this program will process a folder of audio files, extract a comprehensive set of features from each track, and save the results to a CSV file.

def write_batch(buffer, output_csv, fieldnames):
    """Write a batch of rows to CSV at once"""
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    try:
        with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            for row in buffer:
                writer.writerow(row)
        print(f"Wrote {len(buffer)} rows to CSV")
    except Exception as e:
        print(f"WRITE FAILED: {e}")

"""
To Run:

# Default: 4 workers
process_folder(folder_path, output_csv)

# Or customize workers (use your CPU count - 1):
process_folder(folder_path, output_csv, num_workers=6)



"""


if __name__ == "__main__":

    process_folder(

         folder_path=r"G:\AI_Music_Detection_Library\Real_Music2",
         output_csv=r"C:\Users\DOM\Documents\Python\AI_Music_Detection_Software\Human_Output2.csv",
         num_workers=4
    )