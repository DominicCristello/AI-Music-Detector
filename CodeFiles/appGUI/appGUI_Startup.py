import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import sys
import random

# Keep source runs and PyInstaller builds on the same resource layout. In a
# frozen build PyInstaller exposes bundled files beneath sys._MEIPASS; while
# developing, the resource root is the project directory above CodeFiles.
if getattr(sys, 'frozen', False):
    RESOURCE_ROOT = os.path.abspath(sys._MEIPASS)
    CODEFILES_DIR = RESOURCE_ROOT
else:
    CODEFILES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESOURCE_ROOT = os.path.dirname(CODEFILES_DIR)
if CODEFILES_DIR not in sys.path:
    sys.path.insert(0, CODEFILES_DIR)
from computeinfo import compute_info
import threading
from appGUI_Main import AIDetectorApp as MainApp, CURRENT_MODEL, set_music_note_icon
import librosa
import numpy as np

import traceback
import pickle
import re
import pandas as pd
import ctypes

# Use the same fixed scaling policy as LoginScreen.  This block is also needed
# when appGUI_Startup.py is run directly instead of being opened after login.
try:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_widget_scaling(1.0)
    ctk.set_window_scaling(1.0)
except Exception:
    pass

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
# ── Tier (manually set during development; later comes from Stripe plan) ──────
# One of: "basic_model", "premium_model", "business_model"
CURRENT_TIER = CURRENT_MODEL
CURRENT_USER_EMAIL = None
AUTH_TOKEN = None

# ── Config ────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1400, 800
HALF         = WIN_H // 2
SPEC_Y       = 8
SPEC_H       = HALF // 2 - 6
VISUAL_GAP   = 4
WF_Y         = SPEC_Y + SPEC_H + VISUAL_GAP
WF_H         = HALF // 2 - 6
SPEED        = 0.6
FPS_MS       = 16
BTN_Y        = HALF + (WIN_H - HALF) // 2
STARTUP_INK  = (90, 93, 98)  # opaque at roughly half the previous brightness
STARTUP_BG_ALPHA = 178

MOVING_SONGS_DIR = os.path.join(RESOURCE_ROOT, 'MovingSongs')
MOVING_SONG_EXTENSIONS = {
    '.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.aiff', '.aif'
}

# ── Gradient background ───────────────────────────────────────────────────────
def make_gradient(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array([4, 4, 14], dtype=np.float32)
    img  = np.tile(base, (h, w, 1))
    dx  = (xx - w*0.70)/(w*0.45); dy  = (yy - h*0.38)/(h*0.55)
    blue = np.clip(1.0-np.sqrt(dx**2+dy**2),0,1)**2.2
    img[:,:,0]+=blue*12; img[:,:,1]+=blue*35; img[:,:,2]+=blue*130
    dx2 = (xx-w*0.25)/(w*0.38); dy2 = (yy-h*0.78)/(h*0.45)
    purple = np.clip(1.0-np.sqrt(dx2**2+dy2**2),0,1)**1.8
    img[:,:,0]+=purple*90; img[:,:,1]+=purple*8; img[:,:,2]+=purple*110
    dx3 = (xx-w*0.10)/(w*0.30); dy3 = (yy-h*0.15)/(h*0.30)
    teal = np.clip(1.0-np.sqrt(dx3**2+dy3**2),0,1)**2.5
    img[:,:,1]+=teal*30; img[:,:,2]+=teal*60
    return Image.fromarray(np.clip(img,0,255).astype(np.uint8))

def black_white_only(pil_img, threshold=32):
    """Flatten fallback artwork to opaque black and the muted startup white."""
    rgba = np.asarray(pil_img.convert('RGBA'), dtype=np.uint8)
    brightness = np.max(rgba[:, :, :3], axis=2).astype(np.float32)
    brightness *= rgba[:, :, 3].astype(np.float32) / 255.0
    mask = brightness >= threshold
    result = np.zeros((rgba.shape[0], rgba.shape[1], 4), dtype=np.uint8)
    result[:, :, 3] = STARTUP_BG_ALPHA
    result[mask, :3] = STARTUP_INK
    result[mask, 3] = 255
    return Image.fromarray(result)

def make_tile(pil_img, target_h, screen_w, min_width=0):
    ratio  = target_h/pil_img.height
    tile_w = max(int(pil_img.width*ratio),200)
    # NEAREST is intentional: interpolation would introduce extra edge shades
    # beyond the startup artwork's black + muted-white palette.
    resampling = getattr(Image, 'Resampling', Image).NEAREST
    resized = pil_img.convert('RGBA').resize((tile_w, target_h), resampling)
    # Keep one viewport beyond the period so wrapping never exposes an empty
    # edge. min_width lets the shorter spectrogram fill the waveform strip.
    needed = max(screen_w * 2.2, tile_w + screen_w, min_width)
    copies  = max(2, int(np.ceil(needed/tile_w)))
    tiled = Image.new('RGBA', (tile_w*copies, target_h),
                      (0, 0, 0, STARTUP_BG_ALPHA))
    for i in range(copies): tiled.paste(resized,(i*tile_w,0))
    return tiled, tile_w


# ══════════════════════════════════════════════════════════════════════════════
#  Loading overlay — plain tk.Frame placed ON the root (moves with window)
# ══════════════════════════════════════════════════════════════════════════════
class LoadingOverlay:
    RING_BG = '#1A1A2E'
    RING_FG = '#38BDF8'
    OVLY_BG = '#06060F'

    def __init__(self, root, on_cancel=None):
        self.root = root
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()

        # Frame placed directly on root — never detaches when window moves
        self.frame = tk.Frame(root, bg=self.OVLY_BG)
        self.frame.place(x=0, y=0, relwidth=1, relheight=1)
        self.frame.lift()

        # Block all clicks from passing through
        for ev in ('<Button-1>','<Button-2>','<Button-3>',
                   '<ButtonRelease-1>','<B1-Motion>',
                   '<Double-Button-1>'):
            self.frame.bind(ev, lambda e: 'break')

        # Ring — 2x original size. Rendered with PIL (supersampled + downscaled)
        # so the edges are anti-aliased instead of the jagged look tk's canvas
        # arcs give.
        rs = max(520, min(int(min(w,h)*0.80), 840))
        self._rs     = rs
        self._ring_w = max(13, int(rs*0.055))

        self.canvas = tk.Canvas(self.frame, width=rs, height=rs,
                                bg=self.OVLY_BG, highlightthickness=0)
        self.canvas.place(relx=0.5, rely=0.40, anchor='center')

        # Image holds the ring; the % text sits on top of it.
        self._ring_tk = None
        self._ring_item = self.canvas.create_image(rs//2, rs//2, anchor='center')
        fs = max(58, int(rs*0.125))
        self._pct = self.canvas.create_text(rs//2, rs//2, text='0%',
                                             fill='#FFFFFF',
                                             font=('Arial', fs, 'bold'))
        self._render_ring(0.0)

        self._msg = tk.Label(self.frame, text='Initializing...',
                             bg=self.OVLY_BG, fg='#4A6A8A',
                             font=('Segoe UI', 22),
                             wraplength=int(w*0.65))
        self._msg.place(relx=0.5, rely=0.80, anchor='center')

        if on_cancel:
            self._btn = tk.Button(
                self.frame, text='Cancel',
                bg='#252535', fg='#666677',
                activebackground='#CC2222', activeforeground='#FFFFFF',
                relief='flat', bd=0, font=('Segoe UI',20),
                padx=24, pady=8, cursor='hand2', command=on_cancel)
            self._btn.place(relx=0.5, rely=0.90, anchor='center')
            self._btn.bind('<Enter>',
                lambda e: self._btn.configure(bg='#CC2222', fg='#FFFFFF'))
            self._btn.bind('<Leave>',
                lambda e: self._btn.configure(bg='#252535', fg='#666677'))

    @staticmethod
    def _hex_rgb(h):
        return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

    def _render_ring(self, progress):
        # Draw at SSx resolution, then downscale with LANCZOS for smooth edges.
        SS   = 2
        rs   = self._rs
        size = rs * SS
        w    = self._ring_w * SS
        img  = Image.new('RGB', (size, size), self._hex_rgb(self.OVLY_BG))
        draw = ImageDraw.Draw(img)
        pad  = w / 2 + SS * 2
        box  = [pad, pad, size - pad, size - pad]
        # Full background ring, then the coloured progress arc clockwise from top.
        draw.arc(box, 0, 360, fill=self._hex_rgb(self.RING_BG), width=int(w))
        if progress > 0:
            draw.arc(box, -90, -90 + 360.0 * min(1.0, progress),
                     fill=self._hex_rgb(self.RING_FG), width=int(w))
        img = img.resize((rs, rs), Image.LANCZOS)
        self._ring_tk = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._ring_item, image=self._ring_tk)

    def update(self, progress: float, message: str):
        pct = int(progress*100)
        self._render_ring(max(0.0, min(1.0, progress)))
        self.canvas.itemconfig(self._pct, text=f'{pct}%')
        self._msg.config(text=message)
        self.frame.update_idletasks()

    def close(self):
        self.frame.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  Shared feature parser (module-level so appGUI_Main can import it too)
# ══════════════════════════════════════════════════════════════════════════════
_FIELD_MAP = {
    'filename':'fileName','stereo_correlation':'stereoCorrelation',
    'sample_rate':'sampleRate[Hz]','duration':'duration[s]',
    'tempo_estimate':'tempoEstimate[BPM]',
    'rms_mean_mean':'rmsMean[amp]','rms_std_std':'rmsStd[amp]',
    'rms_mean_in_first_5s_mean':'rmsMeanFirst5s[amp]',
    'rms_std_in_first_5s_std':'rmsStdFirst5s[amp]',
    'rms_mean_in_first_30_seconds_mean':'rmsMeanFirst30s[amp]',
    'rms_std_in_first_30_seconds_std':'rmsStdFirst30s[amp]',
    'rms_mean_in_middle_20_seconds_mean':'rmsMeanMid20s[amp]',
    'rms_std_in_middle_20_seconds_std':'rmsStdMid20s[amp]',
    'rms_mean_in_last_20_seconds_mean':'rmsMeanLast20s[amp]',
    'rms_std_in_last_20_seconds_std':'rmsStdLast20s[amp]',
    'zcr_mean_mean':'zcrMean','zcr_std_std':'zcrStd',
    'mean_spectral_centroid_mean':'spectralCentroidMean[Hz]',
    'spectral_centroid_std_std':'spectralCentroidStd[Hz]',
    'mean_spectral_bandwidth_mean':'spectralBandwidthMean[Hz]',
    'spectral_bandwidth_std_std':'spectralBandwidthStd[Hz]',
    'mean_spectral_flatness_mean':'spectralFlatnessMean',
    'mean_spectral_flux_mean':'spectralFluxMean',
    'mean_spectral_rolloff_mean':'spectralRolloffMean[Hz]',
    'ibi_variance':'ibiVariance[s2]',
    'left-right_correlation':'leftRightCorrelation',
    'harmonic_tempo':'harmonicTempo[BPM]',
    'percussive_tempo':'percussiveTempo[BPM]',
    'amplitude_of_harmonic_component':'harmonicAmplitude[amp]',
    'mean_mfcc_mean':'mfccMean','mfcc_std_std':'mfccStd',
    'contrasted_mean_mfcc_mean':'mfccDeltaMean',
    'mean_spectral_contrast_mean':'spectralContrastMean[dB]',
    'mean_chroma_mean':'chromaMean','chroma_variance':'chromaVariance',
    'contrast_variance':'contrastVariance[dB2]',
    'tempo_stability':'tempoStability',
    'standard_deviation_of_tempo':'tempoStd[BPM]',
    'rythym_complexity':'rhythmComplexity',
    'dynamic_range':'dynamicRange[amp]',
    'rythym_entropy':'rhythmEntropy[nats]',
    'skewness':'rmsSkewness','kurtosis':'rmsKurtosis',
    'spectral_peakiness':'spectralPeakiness',
    'phase_incoherence':'phaseIncoherence[rad]',
    'onset_jitter':'onsetJitter[s]',
    'pitch_stability':'pitchStability[Hz]',
    'fine-pitch_fluctuations_std':'finePitchFluctuationsStd[Hz]',
}


def parse_features(info_str: str) -> dict:
    raw = {}
    for line in info_str.splitlines():
        if ': ' not in line: continue
        key, value = line.split(': ', 1)
        bk = key.strip().lower().replace(' ','_').replace('(','').replace(')','')
        if '=' in value:
            for part in value.split(','):
                if '=' not in part: continue
                sk, sv = part.split('=', 1)
                fk  = f"{bk}_{sk.strip().lower()}"
                num = ''.join(c for c in sv if c.isdigit() or c in '.-')
                if num:
                    try: raw[fk] = float(num)
                    except ValueError: pass
        else:
            num = ''.join(c for c in value if c.isdigit() or c in '.-')
            if num:
                try: raw[bk] = float(num)
                except ValueError: pass
    return {_FIELD_MAP.get(k, k): v for k, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  Model inference — turn parsed features into a real P(AI) with model_v2.pkl
# ══════════════════════════════════════════════════════════════════════════════
# The build definition mirrors model_v2.pkl at the resource root, so this path
# is identical in source and packaged runs.
_MODEL_PATH = os.path.join(RESOURCE_ROOT, 'model_v2.pkl')
_MODEL = None
_MODEL_FEATURES = None


def _load_model():
    """Lazy-load the pickled LightGBM model once and cache it, along with the
    exact feature order it was trained on (we drive inference off that)."""
    global _MODEL, _MODEL_FEATURES
    if _MODEL is None:
        with open(_MODEL_PATH, 'rb') as f:
            _MODEL = pickle.load(f)
        try:
            _MODEL_FEATURES = list(_MODEL.feature_name_)
        except Exception:
            _MODEL_FEATURES = list(_MODEL.booster_.feature_name())
    return _MODEL, _MODEL_FEATURES


def _sanitize(name):
    # Same sanitisation train_model_v2.py applied to the column names.
    return re.sub(r'[^0-9a-zA-Z_]', '_', name)


def predict_ai_analysis(features: dict):
    """Return ``(P(AI), per_feature_contributions)``.

    A named DataFrame preserves the feature metadata used during fitting (and
    avoids sklearn's "X does not have valid feature names" warning). LightGBM's
    ``pred_contrib`` values are additive contributions in raw-score/log-odds
    space; the GUI uses only sectional-feature contributions for its waveform.
    """
    model, feat_names = _load_model()
    san = {_sanitize(k): v for k, v in features.items()}
    original_for_safe = {_sanitize(k): k for k in features}
    row = []
    for n in feat_names:
        v = san.get(n, np.nan)
        try:
            row.append(float(v))
        except (TypeError, ValueError):
            row.append(np.nan)
    X = pd.DataFrame([row], columns=feat_names, dtype=float)
    score = float(model.predict_proba(X)[0, 1])

    contrib_row = np.asarray(
        model.booster_.predict(X, pred_contrib=True))[0]
    # Last value is the expected/base value rather than a feature contribution.
    contributions = {}
    for safe_name, contribution in zip(feat_names, contrib_row[:-1]):
        original_name = original_for_safe.get(safe_name, safe_name)
        contributions[original_name] = float(contribution)
    return score, contributions


def predict_ai_score(features: dict) -> float:
    """Backward-compatible score-only wrapper."""
    return predict_ai_analysis(features)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  AnalysisRunner — reusable loading-overlay + background analysis.
#  Used by the startup screen AND by the main GUI's "Upload New Track" button,
#  so the heavy pipeline lives in exactly one place.
# ══════════════════════════════════════════════════════════════════════════════
class AnalysisRunner:
    """
    Shows a LoadingOverlay on `root`, analyzes `audio_path` in a background
    thread, and calls on_done(result_dict) on the main thread when finished.
    on_cancel() is called if the user cancels.  result_dict contains:
        y, sr, D_db, wf_t, wf_y, features, ai_score, y_audio, sr_audio, audio_path
    """
    def __init__(self, root, audio_path, on_done, on_cancelled=None):
        self.root        = root
        self.audio_path  = audio_path
        self.on_done     = on_done
        self.on_cancelled = on_cancelled
        self._cancel     = threading.Event()
        self._hb_job     = None
        self.overlay     = LoadingOverlay(root, on_cancel=self._do_cancel)
        self._heartbeat()
        threading.Thread(target=self._worker, daemon=True).start()

    def _heartbeat(self):
        if self.overlay is not None:
            try: self.root.update()
            except Exception: pass
            self._hb_job = self.root.after(50, self._heartbeat)

    def _stop_heartbeat(self):
        if self._hb_job:
            try: self.root.after_cancel(self._hb_job)
            except Exception: pass
            self._hb_job = None

    def _do_cancel(self):
        self._cancel.set()

    def _worker(self):
        def progress(p, msg):
            if self._cancel.is_set():
                raise InterruptedError('Cancelled')
            self.root.after(0, lambda p=p, msg=msg: self.overlay.update(p, msg))
        try:
            progress(0.01, 'Loading audio file...')
            y_full, sr_full = librosa.load(self.audio_path, sr=44100, mono=False)

            if y_full.ndim == 1:
                y_full = np.stack([y_full, y_full]).astype(np.float32)
            elif y_full.dtype == object:
                ml = min(len(ch) for ch in y_full)
                y_full = np.array([np.array(ch[:ml], dtype=np.float32)
                                   for ch in y_full], dtype=np.float32)
            else:
                y_full = np.array(y_full, dtype=np.float32)
            if y_full.ndim == 1:
                y_full = np.stack([y_full, y_full])
            if y_full.shape[0] > 2:
                y_full = y_full[:2]

            progress(0.02, 'Preparing audio for analysis...')
            info_str = compute_info(y_full, sr_full, self.audio_path,
                                    on_progress=progress)

            progress(0.97, 'Parsing feature values...')
            features = parse_features(info_str)

            progress(0.97, 'Computing spectrogram data...')
            y_mono  = np.mean(y_full, axis=0).astype(np.float32)
            y_disp  = librosa.resample(y_mono, orig_sr=sr_full, target_sr=22050)
            sr_disp = 22050
            D_db    = librosa.amplitude_to_db(
                np.abs(librosa.stft(y_disp, n_fft=1024, hop_length=256)),
                ref=np.max)

            progress(0.98, 'Generating waveform...')
            times = np.linspace(0, len(y_disp)/sr_disp, len(y_disp))
            step  = max(1, len(y_disp)//8000)
            wf_t  = times[::step]
            wf_y  = y_disp[::step]

            progress(0.99, 'Scoring with detection model...')
            try:
                ai_score, feature_contributions = predict_ai_analysis(features)
            except Exception as e:
                traceback.print_exc()
                print(f'[model] scoring failed; showing 0.5. Reason: {e}')
                ai_score = 0.5
                feature_contributions = {}

            progress(1.0, 'Analysis complete!')
            result = dict(
                y=y_disp, sr=sr_disp, D_db=D_db, wf_t=wf_t, wf_y=wf_y,
                features=features, ai_score=ai_score,
                feature_contributions=feature_contributions,
                y_audio=y_full, sr_audio=sr_full,
                audio_path=self.audio_path)
            self.root.after(700, lambda: self._finish(result))

        except InterruptedError:
            self.root.after(0, self._cancelled)
        except Exception as e:
            traceback.print_exc()
            self.root.after(0, lambda: self._errored(str(e)))

    def _finish(self, result):
        self._stop_heartbeat()
        ov = self.overlay
        self.overlay = None
        self.on_done(result, ov)

    def _cancelled(self):
        self._stop_heartbeat()
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        if self.on_cancelled:
            self.on_cancelled()

    def _errored(self, msg):
        self._stop_heartbeat()
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        print(f'[Analysis error] {msg}')
        if self.on_cancelled:
            self.on_cancelled()


# ══════════════════════════════════════════════════════════════════════════════
#  Startup app
# ══════════════════════════════════════════════════════════════════════════════
class AIDetectorApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("AI Music Detector")
        set_music_note_icon(self.root)
        self._heartbeat_job = None
        # Set dark bg immediately so no white/gray flash on startup
        self.root.configure(bg='#040414')

        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        global WIN_W, WIN_H, HALF, VISUAL_GAP
        global SPEC_Y, SPEC_H, WF_Y, WF_H, BTN_Y
        WIN_W  = screen_w; WIN_H  = screen_h
        HALF   = WIN_H // 2
        SPEC_Y = 8
        SPEC_H = HALF // 2 - 6
        VISUAL_GAP = 4
        WF_Y = SPEC_Y + SPEC_H + VISUAL_GAP
        WF_H = HALF // 2 - 6
        BTN_Y  = HALF + (WIN_H-HALF)//2

        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        self.root.resizable(True, True)
        self.root.after(0, lambda: self.root.state('zoomed'))
        self.wf_offset    = 0.0
        self._paused      = False
        self._running     = True
        self._resize_job  = None
        self._overlay     = None
        self._cancel_event = threading.Event()
        self._startup_visuals = None
        self._startup_visual_error = None
        self._startup_visual_ready = threading.Event()
        self._startup_loading_job = None
        self._startup_loading_dots = 0

        self._build_canvas()
        self._build_startup_loading_state()
        self._begin_loading_images()

    # ── Canvas & background ───────────────────────────────────────────────────
    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, width=WIN_W, height=WIN_H,
                                highlightthickness=0, bd=0, bg='#040414')
        self.canvas.place(x=0, y=0)
        try:
            bg_pil = Image.open(os.path.join(
                RESOURCE_ROOT, 'Screenshots_AIMusic', 'background.png'
            )).resize((WIN_W, WIN_H), Image.LANCZOS)
        except Exception:
            bg_pil = make_gradient(WIN_W, WIN_H)
        self._bg_tk = ImageTk.PhotoImage(bg_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self._bg_tk)
    def _heartbeat(self):
        """Keep the main thread responsive to Windows during long computations."""
        if self._overlay is not None:
            # Force Windows to register the app as responsive
            self.root.update()
            self._heartbeat_job = self.root.after(30, self._heartbeat)

    @staticmethod
    def _song_strip_width(duration_seconds):
        """Shared horizontal scale for the full-song waveform and spectrogram."""
        width = max(1800, int(WIN_W * 1.35), int(duration_seconds * 16.0))
        return min(width, 14000)

    def _render_waveform_image(self, samples, duration_seconds, width=None):
        """Render a smooth min/max audio envelope as a reusable PIL image."""
        y = np.asarray(samples, dtype=np.float32).reshape(-1)
        y = y[np.isfinite(y)]
        if y.size < 32:
            raise ValueError('Audio excerpt did not contain enough samples.')

        y = y - float(np.mean(y))
        scale = float(np.percentile(np.abs(y), 99.5))
        if scale < 1e-7:
            raise ValueError('The randomly selected audio excerpt was silent.')
        y = np.clip(y / scale, -1.0, 1.0)

        # Preserve the complete song as a long horizontal strip. Longer songs
        # receive proportionally more horizontal detail, within a practical
        # memory ceiling for the decorative startup animation.
        width = width or self._song_strip_width(duration_seconds)
        height = 360
        if y.size < width:
            old_x = np.linspace(0.0, 1.0, y.size)
            new_x = np.linspace(0.0, 1.0, width)
            y = np.interp(new_x, old_x, y).astype(np.float32)

        # Draw directly in the final two-colour format. Anti-aliasing is avoided
        # because it necessarily creates gray transition pixels.
        image = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        centre = (height - 1) / 2.0
        # RMS keeps dense mastered audio readable; normalizing the envelope by
        # its true maximum below makes the track's highest peak touch both the
        # top and bottom edges of the black waveform panel.
        amplitude = (height - 1) / 2.0

        edges = np.linspace(0, y.size, width + 1, dtype=np.int64)
        waveform_colour = STARTUP_INK
        envelope = np.empty(width, dtype=np.float32)
        for x in range(width):
            start = int(edges[x])
            stop = max(start + 1, int(edges[x + 1]))
            block = y[start:stop]
            envelope[x] = float(np.sqrt(np.mean(block * block)))

        envelope_scale = float(np.max(envelope))
        if envelope_scale > 1e-8:
            envelope = np.clip(envelope / envelope_scale, 0.0, 1.0)
        for x, level in enumerate(envelope):
            lo = centre - float(level) * amplitude
            hi = centre + float(level) * amplitude
            draw.line((x, int(lo), x, int(hi)),
                      fill=waveform_colour, width=1)
        return image

    def _render_spectrogram_image(self, samples, sample_rate,
                                  duration_seconds, width=None):
        """Render the selected song's complete STFT spectrogram to a PIL strip."""
        y = np.asarray(samples, dtype=np.float32).reshape(-1)
        y = y[np.isfinite(y)]
        if y.size < 2048:
            raise ValueError('Audio did not contain enough samples for a spectrogram.')

        width = width or self._song_strip_width(duration_seconds)
        # The startup spectrogram is decorative and ultimately width-limited.
        # Choose a hop that covers the entire song while avoiding an enormous
        # intermediate STFT matrix for long files.
        desired_frames = max(1200, min(width * 2, 18000))
        hop_length = max(256, int(np.ceil(y.size / desired_frames)))
        magnitude = np.abs(librosa.stft(
            y, n_fft=2048, hop_length=hop_length, center=True))
        db = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
        # Strong spectral energy is pure white; everything else is pure black.
        # A -55 dB cutoff preserves musical structure without producing a gray
        # gradient or colour-map values.
        binary = np.where(db >= -55.0, 255, 0).astype(np.uint8)
        # Crop only completely empty frequency rows, then stretch the active
        # range to the full panel. The highest and lowest visible energy meet
        # the black rectangle's boundaries without discarding active data.
        active_rows = np.flatnonzero(np.any(binary > 0, axis=1))
        if active_rows.size:
            binary = binary[active_rows[0]:active_rows[-1] + 1, :]
        # STFT bins run low-to-high from top to bottom in the array; flip them
        # so low frequencies appear at the bottom like the detector graph.
        spectrogram = Image.fromarray(np.flipud(binary))
        resampling = getattr(Image, 'Resampling', Image).NEAREST
        return spectrogram.resize((width, 360), resampling)

    def _random_song_visuals(self):
        """Decode one complete random song and render both synchronized plots."""
        if not os.path.isdir(MOVING_SONGS_DIR):
            raise FileNotFoundError(
                f'MovingSongs directory does not exist: {MOVING_SONGS_DIR}')

        files = [
            os.path.join(MOVING_SONGS_DIR, name)
            for name in os.listdir(MOVING_SONGS_DIR)
            if (os.path.isfile(os.path.join(MOVING_SONGS_DIR, name)) and
                os.path.splitext(name)[1].lower() in MOVING_SONG_EXTENSIONS)
        ]
        if not files:
            raise FileNotFoundError('MovingSongs contains no supported audio files.')

        rng = random.SystemRandom()
        rng.shuffle(files)
        failures = []
        for audio_path in files:
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise ValueError('Invalid audio metadata.')
                data, _sr = sf.read(audio_path, dtype='float32', always_2d=True)
                samples = np.mean(data, axis=1)
                duration = info.frames / float(info.samplerate)
                start_seconds = rng.uniform(0.0, duration) if duration > 0 else 0.0
                strip_width = self._song_strip_width(duration)
                waveform = self._render_waveform_image(
                    samples, duration, width=strip_width)
                spectrogram = self._render_spectrogram_image(
                    samples, info.samplerate, duration, width=strip_width)
                self._moving_song_path = audio_path
                self._moving_song_start = start_seconds
                self._moving_song_fraction = (
                    start_seconds / duration if duration > 0 else 0.0)
                print(f'[startup] moving song visuals: {os.path.basename(audio_path)} '
                      f'@ {start_seconds:.1f}s')
                return waveform, spectrogram
            except Exception as soundfile_error:
                # Compressed formats unsupported by the installed libsndfile
                # build get one decoder fallback through librosa.
                try:
                    try:
                        duration = float(librosa.get_duration(path=audio_path))
                    except TypeError:
                        duration = float(librosa.get_duration(filename=audio_path))
                    samples, _sr = librosa.load(
                        audio_path, sr=22050, mono=True)
                    duration = len(samples) / float(_sr)
                    start_seconds = rng.uniform(0.0, duration) if duration > 0 else 0.0
                    strip_width = self._song_strip_width(duration)
                    waveform = self._render_waveform_image(
                        samples, duration, width=strip_width)
                    spectrogram = self._render_spectrogram_image(
                        samples, _sr, duration, width=strip_width)
                    self._moving_song_path = audio_path
                    self._moving_song_start = start_seconds
                    self._moving_song_fraction = (
                        start_seconds / duration if duration > 0 else 0.0)
                    print(f'[startup] moving song visuals: {os.path.basename(audio_path)} '
                          f'@ {start_seconds:.1f}s')
                    return waveform, spectrogram
                except Exception as librosa_error:
                    failures.append(
                        f'{os.path.basename(audio_path)}: '
                        f'{soundfile_error}; fallback: {librosa_error}')

        raise RuntimeError('Could not decode a MovingSongs file. ' + ' | '.join(failures))

    # ── Image loading ─────────────────────────────────────────────────────────
    def _build_startup_loading_state(self):
        """Keep the new window informative while its decorative audio is built."""
        self._startup_loading_title = self.canvas.create_text(
            WIN_W // 2, WIN_H // 2 - 18,
            text='Preparing your workspace', fill='#FFFFFF',
            font=('Segoe UI', 27, 'bold'), anchor='center',
            tags=('startup_loading',))
        self.canvas.create_text(
            WIN_W // 2, WIN_H // 2 + 28,
            text='Decoding the startup audio visualization', fill='#6A9AB0',
            font=('Segoe UI', 14), anchor='center',
            tags=('startup_loading',))
        self._animate_startup_loading()

    def _animate_startup_loading(self):
        if self._startup_visual_ready.is_set():
            return
        self._startup_loading_dots = (self._startup_loading_dots + 1) % 4
        try:
            self.canvas.itemconfigure(
                self._startup_loading_title,
                text='Preparing your workspace' +
                     '.' * self._startup_loading_dots)
        except tk.TclError:
            return
        self._startup_loading_job = self.root.after(
            320, self._animate_startup_loading)

    def _begin_loading_images(self):
        """Decode/render without blocking the Tk event loop or blanking the UI."""
        threading.Thread(target=self._load_images_worker, daemon=True).start()
        self.root.after(50, self._poll_loaded_images)

    def _load_images_worker(self):
        sp_path = os.path.join(
            RESOURCE_ROOT, 'Screenshots_AIMusic',
            'spectrogram_white_transparent.png')
        wf_path = os.path.join(
            RESOURCE_ROOT, 'Screenshots_AIMusic',
            'waveform_white_transparent.png')
        try:
            try:
                waveform, spectrogram = self._random_song_visuals()
            except Exception as exc:
                # The startup page must remain usable if the sample directory
                # is missing, empty, or contains an unreadable file.
                print('[startup] random song visuals unavailable; '
                      f'using fallback: {exc}')
                # Copy pixels while the files are open so no lazy file handle
                # crosses back to Tk's main thread.
                with Image.open(wf_path) as fallback_waveform:
                    waveform = fallback_waveform.copy()
                with Image.open(sp_path) as fallback_spectrogram:
                    spectrogram = fallback_spectrogram.copy()
                self._moving_song_path = None
                self._moving_song_start = None
                self._moving_song_fraction = 0.0

            self._startup_visuals = (
                black_white_only(spectrogram),
                black_white_only(waveform),
            )
        except Exception as exc:
            self._startup_visual_error = exc
        finally:
            self._startup_visual_ready.set()

    def _poll_loaded_images(self):
        if not self._startup_visual_ready.is_set():
            self.root.after(50, self._poll_loaded_images)
            return

        if self._startup_loading_job is not None:
            try:
                self.root.after_cancel(self._startup_loading_job)
            except (tk.TclError, ValueError):
                pass
            self._startup_loading_job = None

        if self._startup_visual_error is not None:
            print(f'[startup] could not prepare visuals: '
                  f'{self._startup_visual_error}')
            self.canvas.itemconfigure(
                self._startup_loading_title,
                text='Unable to prepare the startup visualization')
            return

        self._sp_pil, self._wf_pil = self._startup_visuals
        self._startup_visuals = None
        self._make_tiles()
        self.canvas.delete('startup_loading')
        self._build_ui()
        self._animate()
        self.root.bind('<Configure>', self._on_resize)

    def _make_tiles(self):
        old_period = getattr(self, '_combined_tw', 0)
        if old_period:
            phase = (self.wf_offset / old_period) % 1.0
        else:
            phase = getattr(self, '_moving_song_fraction', 0.0)

        wf_tile, self._wf_tw = make_tile(self._wf_pil, WF_H, WIN_W)
        sp_tile, self._sp_tw = make_tile(
            self._sp_pil, SPEC_H, WIN_W, min_width=wf_tile.width)
        tile_w   = max(sp_tile.width, wf_tile.width)
        local_gap = WF_Y - SPEC_Y - SPEC_H
        combined = Image.new(
            'RGBA', (tile_w, SPEC_H + local_gap + WF_H),
            (0, 0, 0, STARTUP_BG_ALPHA))
        combined.paste(sp_tile,(0,0))
        combined.paste(wf_tile, (0, SPEC_H + local_gap))
        self._combined_tile = combined
        # The waveform period represents one complete song, so animation wraps
        # only after traversing that full waveform—not at the shorter artwork.
        self._combined_tw   = self._wf_tw
        self.wf_offset      = phase * self._combined_tw
        self._combined_tk   = ImageTk.PhotoImage(combined)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._combined_item = self.canvas.create_image(
            0, SPEC_Y, anchor='nw', image=self._combined_tk)
        self.canvas.create_line(WIN_W*0.25, HALF+2, WIN_W*0.75, HALF+2,
                                fill='#1E2E40', width=1)

        self._upload_btn = ctk.CTkButton(
            self.root, text='Upload Track', command=self._upload,
            width=240, height=62,
            font=ctk.CTkFont(family='Helvetica', size=20, weight='bold'),
            fg_color='#38BDF8', hover_color='#0EA5E9',
            text_color='#0C1A2E', corner_radius=18, border_width=0)
        self.canvas.create_window(WIN_W//2, BTN_Y, anchor='center',
                                  window=self._upload_btn)

        # Mixed-style format text drawn directly on canvas (no Frame = no box)
        items = [
            ('supports ', '#FFFFFF', False),
            ('.mp3',  '#00FF88', True),  (', ', '#00FF88', False),
            ('.wav',  '#00FF88', True),  (', ', '#00FF88', False),
            ('.flac', '#00FF88', True),  (', ', '#00FF88', False),
            ('.ogg',  '#00FF88', True),
        ]
        widths = [tkfont.Font(family='Arial', size=12,
                  underline=ul).measure(t) for t, _, ul in items]
        x = WIN_W//2 - sum(widths)//2
        for (text, color, underline), w in zip(items, widths):
            font = ('Arial', 12, 'underline') if underline else ('Arial', 12)
            self.canvas.create_text(x, BTN_Y+75, text=text,
                                    font=font, fill=color, anchor='w')
            x += w

    # ── Animation ─────────────────────────────────────────────────────────────
    def _animate(self):
        if not self._running:
            return
        if self._paused:
            # Reschedule much less frequently while paused — saves CPU during drag
            self.root.after(150, self._animate)
            return
        self.wf_offset = (self.wf_offset+SPEED) % self._combined_tw
        self.canvas.coords(self._combined_item, -int(self.wf_offset), SPEC_Y)
        self.root.after(FPS_MS, self._animate)

    def _on_resize(self, event):
        if event.widget != self.root: return
        if self.root.state() == 'iconic': return
        if self._overlay is not None:
            return
        self._paused = True
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(400, self._resume)
        if event.width != WIN_W or event.height != WIN_H:
            if hasattr(self, '_rebuild_job') and self._rebuild_job:
                self.root.after_cancel(self._rebuild_job)
            self._rebuild_job = self.root.after(500, self._rebuild)
    def _resume(self):
        # Only unpause if no loading overlay is active
        if self._overlay is None:
            self._paused = False

    def _rebuild(self):
        # Safety: if widgets were destroyed (main GUI took over), bail out
        if not self._running or not self.canvas.winfo_exists():
            return
        new_w = self.root.winfo_width()
        new_h = self.root.winfo_height()
        if new_w < 200 or new_h < 200: return
        global WIN_W, WIN_H, HALF, VISUAL_GAP
        global SPEC_Y, SPEC_H, WF_Y, WF_H, BTN_Y
        WIN_W=new_w; WIN_H=new_h; HALF=WIN_H//2
        SPEC_Y = 8
        SPEC_H = HALF // 2 - 6
        VISUAL_GAP = 4
        WF_Y = SPEC_Y + SPEC_H + VISUAL_GAP
        WF_H = HALF // 2 - 6
        BTN_Y=HALF+(WIN_H-HALF)//2
        self.canvas.config(width=WIN_W, height=WIN_H)
        self.canvas.delete('all')
        try:
            bg_pil = Image.open(os.path.join(
                RESOURCE_ROOT, 'Screenshots_AIMusic', 'background.png'
            )).resize((WIN_W, WIN_H), Image.LANCZOS)
        except Exception:
            bg_pil = make_gradient(WIN_W, WIN_H)
        self._bg_tk = ImageTk.PhotoImage(bg_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self._bg_tk)
        self._make_tiles()
        self._build_ui()

    # ── Upload ────────────────────────────────────────────────────────────────
    def _upload(self):
        self._paused = True   # freeze animation before dialog opens
        path = filedialog.askopenfilename(
            title='Select Audio Track',
            filetypes=[('Audio Files','*.mp3 *.wav *.flac *.ogg *.aac *.m4a'),
                       ('All Files','*.*')])
        if not path:
            self._paused = False
            return
        self._audio_path = path
        self._upload_btn.configure(
            text='Loading...', fg_color='#2A2A2A',
            hover_color='#2A2A2A', text_color='#888888')
        # Delegate to the shared AnalysisRunner
        self._runner = AnalysisRunner(
            self.root, path,
            on_done=self._on_analysis_done,
            on_cancelled=self._restore_after_cancel)

    def _on_analysis_done(self, result, overlay):
        """Called by AnalysisRunner when analysis finishes."""
        self._launch_main(result, overlay)

    def _restore_after_cancel(self):
        if self._overlay:
            self._overlay.close()
            self._overlay = None
        self._upload_btn.configure(
            text='Upload Track', fg_color='#38BDF8',
            hover_color='#0EA5E9', text_color='#0C1A2E')
        self._paused = False

    # ── Launch main GUI in same window ────────────────────────────────────────
    def _launch_main(self, result, overlay):
        self._running = False
          # Stop the startup resize handlers — main GUI has its own
        try:
            self.root.unbind('<Configure>')
        except Exception:
            pass
        if self._resize_job:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
            self._resize_job = None
        if hasattr(self, '_rebuild_job') and self._rebuild_job:
            try:
                self.root.after_cancel(self._rebuild_job)
            except Exception:
                pass
            self._rebuild_job = None

        root = self.root
        root.configure(bg='#060614')

        # Keep overlay alive — destroy everything ELSE
        overlay_frame = overlay.frame if overlay else None
        for w in root.winfo_children():
            if w is not overlay_frame:
                w.destroy()
        root.update_idletasks()

        # Build main GUI underneath the overlay
        main = MainApp(audio_path=result['audio_path'],
                       y=result['y'], sr=result['sr'], D_db=result['D_db'],
                       wf_t=result['wf_t'], wf_y=result['wf_y'],
                       features=result['features'], ai_score=result['ai_score'],
                       feature_contributions=result.get('feature_contributions'),
                       y_audio=result['y_audio'], sr_audio=result['sr_audio'],
                       root=root, tier=CURRENT_TIER,
                       current_user_email=CURRENT_USER_EMAIL,
                       auth_token=AUTH_TOKEN)

        # Close the loading overlay shortly after the canvas first draws; the
        # plot's own cover keeps hiding it until fully rendered.
        def _on_drawn(event=None):
            if not hasattr(_on_drawn, '_fired'):
                _on_drawn._fired = True
                if overlay:
                    root.after(400, overlay.close)

        if main._mpl_canvas:
            tkw = main._mpl_canvas.get_tk_widget()
            tkw.bind('<Map>', _on_drawn, add='+')
            main._mpl_canvas.mpl_connect('draw_event', _on_drawn)
        else:
            # basic tier may have no spectrogram canvas — close overlay directly
            root.after(400, lambda: overlay.close() if overlay else None)

        # Hard fallback
        root.after(3000, _on_drawn)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = AIDetectorApp()
    app.run()
