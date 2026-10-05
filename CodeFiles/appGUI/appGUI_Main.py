import os
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter
import librosa.display

import customtkinter as ctk

from StoreHistory import HistoryStore, SettingsStore


def set_music_note_icon(window):
    """Replace Tk's default app tile with a clearly legible music note."""
    try:
        from PIL import Image, ImageDraw, ImageTk
        size = 64
        scale = 4
        canvas_size = size * scale
        img = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # No background tile: Windows reduced the old design to what looked
        # like a blue square. This oversized eighth-note silhouette remains
        # recognizable after title-bar downscaling.
        note = (88, 205, 255, 255)
        draw.rounded_rectangle(
            [30 * scale, 38 * scale, 47 * scale, 54 * scale],
            radius=7 * scale, fill=note)
        draw.rectangle(
            [43 * scale, 10 * scale, 49 * scale, 45 * scale], fill=note)
        draw.polygon([
            (43 * scale, 10 * scale),
            (59 * scale, 7 * scale),
            (59 * scale, 16 * scale),
            (49 * scale, 19 * scale),
            (43 * scale, 19 * scale),
        ], fill=note)

        resampling = getattr(Image, 'Resampling', Image).LANCZOS
        icon = ImageTk.PhotoImage(img.resize((size, size), resampling))
        window._music_note_icon = icon  # Tk requires a persistent reference.
        if hasattr(window, '_iconbitmap_method_called'):
            window._iconbitmap_method_called = True
        window.iconphoto(True, icon)
        window.after(300, lambda: window.winfo_exists() and
                     window.iconphoto(True, window._music_note_icon))
    except Exception as exc:
        print(f'[ui] could not set music-note window icon: {exc}')


# ── Fonts & colours ───────────────────────────────────────────────────────────
FONT_BTN  = ('Segoe UI', 11)
FONT_MONO = ('Consolas', 10)
BG_MAIN   = '#060614'
BG_CARD   = '#090918'
BG_BTN    = '#141428'
BG_BTN_H  = '#1E1E38'

# Upgrade button (header, top-right): gray box + lighter-gray text by default,
# glows blue with white text on hover.
UPGRADE_BG   = '#26262E'
UPGRADE_FG   = '#8A94A6'
UPGRADE_BG_H = '#2F7BEE'
UPGRADE_FG_H = '#FFFFFF'

# ── Tier capabilities ──────────────────────────────────────────────────────────
TIER_CAPS = {
    'basic_model': {
        'spectrogram': False, 'playback': False, 'dropdown': True,
        'history': False, 'background': False, 'export': False,
        'max_duration': 600,
    },
    'premium_model': {
        'spectrogram': True, 'playback': False, 'dropdown': True,
        'history': True, 'background': False, 'export': False,
        'max_duration': 1200,
    },
    'business_model': {
        'spectrogram': True, 'playback': True, 'dropdown': True,
        'history': True, 'background': True, 'export': True,
        'max_duration': 1200,
    },
    # Enterprise = the max tier; full feature set, no upgrade path.
    'enterprise_model': {
        'spectrogram': True, 'playback': True, 'dropdown': True,
        'history': True, 'background': True, 'export': True,
        'max_duration': 1200,
    },
}

# ── Plan display (header label text + colour) ──────────────────────────────────
PLAN_DISPLAY = {
    'basic_model':      ('Basic',      '#FFFFFF'),   # white
    'premium_model':    ('Premium',    '#38BDF8'),   # light blue
    'business_model':   ('Business',   '#7C3AED'),   # darker purple
    'enterprise_model': ('Enterprise', '#FF8C00'),   # orange
}

# Runtime source of truth for plan changes made from the Change Plan screen.
CURRENT_MODEL = 'business_model'

# ── Gauge palette ──────────────────────────────────────────────────────────────
def _lerp(c1, c2, t):
    return tuple(int(c1[i] + t*(c2[i]-c1[i])) for i in range(3))

def gauge_color_hex(t):
    R=(220,45,45); Y=(255,200,0); G=(45,205,45)
    rgb = _lerp(R,Y,t*2) if t<=0.5 else _lerp(Y,G,(t-0.5)*2)
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def score_color(ai_score):
    """Colour shared by the gauge percentage and AI-likelihood value."""
    return ('#DC3232' if float(ai_score) > 0.65 else
            '#CCCC22' if float(ai_score) > 0.35 else '#32C832')


def confidence_color(confidence_pct):
    """Colour scale used anywhere classification confidence is displayed."""
    if confidence_pct < 50:
        return '#FF4444'
    if confidence_pct < 75:
        return '#E5C62A'
    if confidence_pct < 90:
        return '#4CD964'
    return '#00FF66'


def classification_for_score(ai_score):
    """Return the user-facing interpretation and colour for P(AI)."""
    pct = float(ai_score) * 100.0
    if pct < 30:
        return 'Definitely Human', '#32C832'
    if pct < 40:
        return 'Probably Human', '#66D978'
    if pct < 60:
        return 'Possibly Human, possibly AI generated', '#E5C62A'
    if pct < 70:
        return ('Likely AI generated (or has very distinctive AI features)',
                '#FF9418')
    if pct < 85:
        return 'Definitely AI generated', '#FF573D'
    return 'Incredibly AI generated', '#FF2020'

# ── Feature classification ────────────────────────────────────────────────────
_RULES = {
    'stereoCorrelation':(0.98,0.93,True),'onsetJitter[s]':(0.005,0.013,False),
    'pitchStability[Hz]':(0.4,1.5,False),'ibiVariance[s2]':(0.001,0.006,False),
    'phaseIncoherence[rad]':(0.04,0.12,False),'rhythmComplexity':(0.001,0.008,False),
    'tempoStability':(0.001,0.004,False),'dynamicRange[amp]':(0.04,0.14,False),
    'spectralPeakiness':(3.2,2.2,True),'rmsSkewness':(0.08,0.4,False),
    'harmonicAmplitude[amp]':(0.05,0.12,False),'leftRightCorrelation':(0.97,0.92,True),
    'finePitchFluctuationsStd[Hz]':(0.5,2.0,False),
}
def classify(name, val):
    try: v=float(val)
    except: return 'neutral'
    if name not in _RULES: return 'neutral'
    r,y,hi=_RULES[name]
    return ('red' if(v>r if hi else v<r) else
            'yellow' if(v>y if hi else v<y) else 'green')
CMAP={'red':'#FF4444','yellow':'#CCCC22','green':'#44CC44','neutral':'#777777'}

# Time-local feature roster taken from feature_separation.csv. These are limited
# to measurements compute_info() calculates for a real section of the song;
# whole-track features must not be presented as if they identify a timestamp.
# The stored means/effect sizes document why each feature was selected, while
# the live verdict uses the trained LightGBM model's own contribution values.
_LOCAL_EVIDENCE = {
    'Intro': [
        ('spectralContrastMeanFirst2s[dB]', 20.999778, 18.535583, .703524),
        ('spectralFlatnessStdFirst2s',       .131506,   .215891, .527122),
        ('spectralFlatnessMeanFirst2s',      .085463,   .199786, .498932),
        ('rmsStdFirst30s[amp]',              .051403,   .068083, .477573),
        ('spectralFluxMeanFirst2s',         1.070745,   .973039, .257700),
        ('spectralContrastStdFirst2s[dB]', 13.632797, 12.637208, .257213),
    ],
    'Middle': [
        ('rmsMeanMid20s[amp]',               .147569,   .190496, .533995),
        ('rmsStdMid20s[amp]',                .050822,   .066454, .476471),
        ('spectralContrastMeanMidSong[dB]', 23.173681, 22.707731, .354771),
        ('spectralRolloffMeanMidSong[Hz]', 6630.9861, 6037.7712, .237862),
        ('spectralCentroidMeanMidSong[Hz]',3073.9953, 2869.4145, .188297),
    ],
    'Ending': [
        ('spectralContrastMeanEndSong[dB]', 16.806190, 14.207890, .579608),
        ('rmsStdLast20s[amp]',                .063170,   .081339, .420806),
        ('chromaMeanEndSong',                 .423231,   .332519, .359055),
        ('spectralContrastStdEndSong[dB]',    9.308810,  7.538531, .340826),
        ('zcrMeanLast20s',                    .056049,   .063059, .204550),
        ('spectralRolloffMeanEndSong[Hz]', 6612.8581, 6124.7494, .195500),
    ],
}

_LOCAL_VERDICTS = {
    'red':    ('Strongly AI', '#FF4444'),
    'yellow': ('Possibly AI', '#E5C62A'),
    'green':  ('Authentic',   '#44CC44'),
    'neutral':('Inconclusive', '#40E0D0'),
}

# History score colours
def history_score_color(score_pct):
    if score_pct < 30:  return '#32C832'   # green
    if score_pct <= 75: return '#CCCC22'   # yellow
    return '#DC3232'                        # red


# ══════════════════════════════════════════════════════════════════════════════
#  AudioEngine
# ══════════════════════════════════════════════════════════════════════════════
class AudioEngine:
    _HEADROOM_DB = -1.0

    def __init__(self, audio, samplerate):
        self._sr   = int(samplerate)
        self._data = self._prepare(audio)
        self._frame   = 0
        self._playing = False
        self._stream  = None
        self._end_cb  = None
        self._intentional_stop = False

    def _prepare(self, audio):
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            a = np.column_stack([a, a])
        elif a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
            a = a.T
        elif a.ndim == 2 and a.shape[1] == 1:
            a = np.column_stack([a[:,0], a[:,0]])
        elif a.ndim == 2 and a.shape[1] > 2:
            a = a[:, :2]
        peak = np.max(np.abs(a)) if a.size else 0.0
        if peak > 1e-8:
            target = 10 ** (self._HEADROOM_DB / 20.0)
            a = a * (target / peak)
        return np.ascontiguousarray(a, dtype=np.float32)

    @property
    def position(self): return self._frame / self._sr
    @property
    def duration(self): return len(self._data) / self._sr
    @property
    def playing(self):  return self._playing

    def set_end_callback(self, cb): self._end_cb = cb

    def _callback(self, outdata, frames, time_info, status):
        import sounddevice as sd
        start = self._frame
        end   = min(start + frames, len(self._data))
        n     = end - start
        outdata[:n] = self._data[start:end]
        self._frame = end
        if n < frames:
            outdata[n:] = 0
            raise sd.CallbackStop()

    def _on_stream_finished(self):
        self._playing = False
        if not self._intentional_stop and self._end_cb:
            self._end_cb()
        self._intentional_stop = False

    def _open(self):
        import sounddevice as sd
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
        self._stream = sd.OutputStream(
            samplerate=self._sr, channels=2, dtype='float32',
            blocksize=512, latency='low',
            callback=self._callback,
            finished_callback=self._on_stream_finished)

    def play(self, from_sec=None):
        if from_sec is not None:
            self._frame = int(max(0.0, min(from_sec, self.duration)) * self._sr)
        if self._frame >= len(self._data):
            self._frame = 0
        self._open()
        self._playing = True
        self._stream.start()

    def pause(self):
        self._intentional_stop = True
        if self._stream:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
            self._stream = None
        self._playing = False

    def stop(self):
        self.pause()
        self._frame = 0

    def close(self):
        self.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Gauge meter
# ══════════════════════════════════════════════════════════════════════════════
class GaugeMeter(tk.Canvas):
    N_SEG = 60
    ARC_W = 28

    def __init__(self, parent, size=700, **kwargs):
        self.R   = int(size * 0.40)
        self.RN  = int(size * 0.37)
        self.CW  = size + 200
        self.CX  = self.CW // 2
        self.CY  = self.R + self.ARC_W // 2 + 10
        self.CH  = self.CY + 40
        super().__init__(parent, width=self.CW, height=self.CH,
                         bg=BG_CARD, highlightthickness=0, **kwargs)
        self._angle  = 180.0
        self._target = 180.0
        self._draw_bg(); self._draw_segs(); self._draw_ticks()
        self._draw_endpoint_mask(); self._draw_labels()
        self._draw_needle(self._angle)

    def _bbox(self, r): return (self.CX-r, self.CY-r, self.CX+r, self.CY+r)

    def _draw_bg(self):
        self.create_arc(*self._bbox(self.R), start=0, extent=180,
                        style='arc', outline='#151525', width=self.ARC_W+10)
        for dr in (self.ARC_W//2+5, -self.ARC_W//2-5):
            self.create_arc(*self._bbox(self.R+dr), start=0, extent=180,
                            style='arc', outline='#1A1A30', width=1)

    def _draw_segs(self):
        seg = 180.0/self.N_SEG
        for i in range(self.N_SEG):
            t = i/(self.N_SEG-1)
            self.create_arc(*self._bbox(self.R), start=i*seg, extent=seg+0.8,
                            style='arc', outline=gauge_color_hex(t),
                            width=self.ARC_W)

    def _draw_ticks(self):
        for deg in range(0, 181, 18):
            rad   = math.radians(deg)
            r_out = self.R + self.ARC_W//2 + 3
            r_in  = self.R - self.ARC_W//2 - 16
            self.create_line(self.CX+r_out*math.cos(rad),
                             self.CY-r_out*math.sin(rad),
                             self.CX+r_in *math.cos(rad),
                             self.CY-r_in *math.sin(rad),
                             fill='#FFFFFF', width=1)
            pct = round((1-deg/180)*100)
            lr  = self.R - self.ARC_W//2 - 36
            y_lift = -14 if deg in (0, 180) else 0
            self.create_text(self.CX+lr*math.cos(rad),
                             self.CY-lr*math.sin(rad)+y_lift,
                             text=f'{pct}%', fill='#555577',
                             font=('Segoe UI', 11, 'bold'))

    def _draw_endpoint_mask(self):
        # bleed = how far below the centre line the mask covers. Increase this
        # if a coloured arc edge shows through the HUMAN / AI labels.
        bleed = self.ARC_W // 2 + 12
        self.create_rectangle(0, self.CY+1, self.CW, self.CY+bleed,
                               fill=BG_CARD, outline='')

    def _draw_labels(self):
        self.create_text(self.CX - self.R, self.CY + 22,
                         text='HUMAN', fill='#32C832',
                         font=('Segoe UI', 12, 'bold'), anchor='center')
        self.create_text(self.CX + self.R, self.CY + 22,
                         text='AI', fill='#DC3232',
                         font=('Segoe UI', 12, 'bold'), anchor='center')

    def _needle_pts(self, deg):
        rad = math.radians(deg)
        tx  = self.CX + self.RN*math.cos(rad)
        ty  = self.CY - self.RN*math.sin(rad)
        b1, b2 = math.radians(deg+7), math.radians(deg-7)
        base = 16
        return (tx, ty,
                self.CX+base*math.cos(b1), self.CY-base*math.sin(b1),
                self.CX+base*math.cos(b2), self.CY-base*math.sin(b2))

    def _draw_needle(self, deg):
        self.delete('needle')
        tx,ty,bx1,by1,bx2,by2 = self._needle_pts(deg)
        self.create_polygon(tx,ty,bx1,by1,bx2,by2, fill='#EEEEEE',
                            outline='#AAAAAA', width=1, tags='needle')
        hr = 12
        self.create_oval(self.CX-hr,self.CY-hr,self.CX+hr,self.CY+hr,
                         fill='#888899', outline='#CCCCCC', width=2,
                         tags='needle')

    def _step(self):
        diff = self._target - self._angle
        if abs(diff) < 0.4:
            self._angle = self._target; self._draw_needle(self._angle); return
        step = max(min(diff*0.07, 4.0), -4.0)
        if abs(step) < 0.25: step = 0.25*(1 if diff>0 else -1)
        self._angle += step; self._draw_needle(self._angle)
        self.after(14, self._step)

    def animate_to(self, ai_score):
        self._angle  = 180.0
        self._target = 180.0*(1.0-float(ai_score))
        self._draw_needle(self._angle)
        self.after(80, self._step)


# ══════════════════════════════════════════════════════════════════════════════
#  Main results app
# ══════════════════════════════════════════════════════════════════════════════
class AIDetectorApp:
    def __init__(self, audio_path, y, sr, D_db, wf_t, wf_y, features, ai_score,
                 y_audio=None, sr_audio=None, root=None, tier='basic_model',
                 from_history=False, feature_contributions=None,
                 current_user_email=None, auth_token=None):
        ctk.set_appearance_mode('dark')

        self.tier = tier if tier in TIER_CAPS else 'basic_model'
        global CURRENT_MODEL
        CURRENT_MODEL = self.tier
        self.caps = TIER_CAPS[self.tier]
        self._from_history  = from_history   # loaded from history → don't re-save
        self._already_saved = False

        if root is not None:
            self.root = root
            self._sw  = root.winfo_screenwidth()
            self._sh  = root.winfo_screenheight()
            self._owns_root = False
        else:
            self.root = ctk.CTk()
            self.root.update_idletasks()
            self._sw = self.root.winfo_screenwidth()
            self._sh = self.root.winfo_screenheight()
            self.root.geometry(f'{self._sw}x{self._sh}')
            self.root.state('zoomed')
            self.root.resizable(True, True)
            self._owns_root = True

        set_music_note_icon(self.root)

        self.audio_path = audio_path
        self.y      = y
        self.sr     = sr
        self.D_db   = D_db
        self.wf_t   = wf_t
        self.wf_y   = wf_y
        self.features  = features
        self.ai_score  = ai_score
        self.feature_contributions = feature_contributions or {}
        self.current_user_email = current_user_email or 'No authenticated account'
        self.auth_token = auth_token
        self.y_audio   = y_audio
        self.sr_audio  = sr_audio
        self._fig_refs = []
        self.dropdown_open = False

        # History persistence
        self.history = HistoryStore()
        self._history_open = False
        self._history_panel = None
        self._history_anim_job = None
        self._history_anim_serial = 0

        # Playback (only used in business tier)
        if self.caps['playback']:
            audio_src = y_audio if y_audio is not None else y
            sr_src    = sr_audio if sr_audio is not None else sr
            self._engine = AudioEngine(audio_src, sr_src)
            self._engine.set_end_callback(
                lambda: self.root.after(0, self._on_track_end))
            self._duration = self._engine.duration
        else:
            self._engine  = None
            self._duration = len(y) / sr

        self._playing  = False
        self._dragging = False
        self._pos      = 0.0
        self._alive    = True   # set False when this instance is torn down
        self._vline_spec = None
        self._vline_wave = None
        self._triangle   = None
        self._mpl_canvas = None
        self._mpl_bg     = None
        self._fig        = None
        self._ax_spec    = None
        self._ax_wave    = None
        self._wave_regions = []
        self._wave_tooltip = None
        self._upgrade_overlay = None   # in-place upgrade page overlay
        self._plan_change_in_progress = False
        self._ui_buttons = []          # locked while the plot is still rendering
        self._ui_unlocked = False

        self._build_layout()

        if self.caps['playback']:
            self.root.bind('<space>', self._on_space)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        # NOTE: history save now happens right after the gauge animation
        # starts (see _build_layout), which is the moment everything is
        # fully rendered.

    # ── Explainable, time-local waveform evidence ──────────────────────────────
    @staticmethod
    def _display_feature_name(name):
        """Compact names for the waveform hover card."""
        replacements = {
            'spectralContrast': 'Spectral contrast ',
            'spectralFlatness': 'Spectral flatness ',
            'spectralRolloff': 'Spectral rolloff ',
            'spectralCentroid': 'Spectral centroid ',
            'spectralFlux': 'Spectral flux ',
            'rmsMean': 'RMS mean ', 'rmsStd': 'RMS variation ',
            'zcrMean': 'Zero-crossing rate ', 'chromaMean': 'Chroma mean ',
            'First2s': '(first 2 s)', 'First30s': '(first 30 s)',
            'Mid20s': '(middle 20 s)', 'MidSong': '(mid-song)',
            'Last20s': '(last 20 s)', 'EndSong': '(ending)',
        }
        text = name
        for old, new in replacements.items():
            text = text.replace(old, new)
        return ' '.join(text.replace('[', ' (').replace(']', ')').split())

    def _build_wave_regions(self):
        """Score intro/middle/ending using the trained model's contributions.

        Each region remains anchored to the full-track probability. Sectional
        evidence can move the regional verdict up/down, but cannot overpower the
        complete 115-feature decision and produce a contradictory visualization.
        """
        bounds = ((0.0, self._duration * .25),
                  (self._duration * .25, self._duration * .75),
                  (self._duration * .75, self._duration))
        regions = []
        for (region_name, specs), (start, end) in zip(_LOCAL_EVIDENCE.items(), bounds):
            evidence = []
            local_sum = 0.0
            for key, _ai_mean, _human_mean, _effect_size in specs:
                if key not in self.features:
                    continue
                try:
                    value = float(self.features[key])
                except (TypeError, ValueError):
                    continue
                contribution = float(self.feature_contributions.get(key, 0.0))
                local_sum += contribution
                evidence.append((abs(contribution), key, value, contribution))

            evidence.sort(reverse=True)
            regions.append({
                'name': region_name, 'start': start, 'end': end,
                'local_sum': local_sum, 'evidence': evidence[:4],
            })

        p = max(1e-6, min(1.0 - 1e-6, float(self.ai_score)))
        overall_logit = math.log(p / (1.0 - p))
        # Width-weighted centering keeps the time-average regional evidence tied
        # to the full model result (middle occupies half of the waveform).
        weighted_local = (regions[0]['local_sum'] * .25 +
                          regions[1]['local_sum'] * .50 +
                          regions[2]['local_sum'] * .25)
        have_contrib = bool(self.feature_contributions)
        for region in regions:
            adjustment = region['local_sum'] - weighted_local if have_contrib else 0.0
            adjustment = max(-1.25, min(1.25, adjustment))
            region_p = 1.0 / (1.0 + math.exp(-(overall_logit + adjustment)))
            region['score'] = region_p
            meaningful = [item for item in region['evidence']
                          if item[0] >= .01]
            evidence_strength = sum(item[0] for item in meaningful)

            # Conservative regional thresholds:
            #   red    = very high score AND positive local model evidence
            #   yellow = plausible AI evidence, but below the strong threshold
            #   green  = low score AND negative/authentic local evidence
            #   turquoise = too little local evidence or an ambiguous middle range
            if len(meaningful) < 2 or evidence_strength < .06:
                verdict = 'neutral'
            elif region_p >= .82 and adjustment >= .10:
                verdict = 'red'
            elif region_p >= .45 and adjustment >= .02:
                verdict = 'yellow'
            elif region_p <= .28 and adjustment <= -.02:
                verdict = 'green'
            else:
                verdict = 'neutral'
            region['verdict'] = verdict
        self._wave_regions = regions
        return regions

    def _plot_forensic_waveform(self, ax):
        """Plot one continuous waveform, colored by local forensic evidence."""
        regions = self._build_wave_regions()
        # A faint base prevents tiny visual gaps at region boundaries.
        ax.plot(self.wf_t, self.wf_y, color='#23383A', linewidth=.7, alpha=.55)
        for region in regions:
            mask = ((self.wf_t >= region['start']) &
                    (self.wf_t <= region['end']))
            if not np.any(mask):
                continue
            color = _LOCAL_VERDICTS[region['verdict']][1]
            ax.plot(self.wf_t[mask], self.wf_y[mask], color=color,
                    linewidth=.8, alpha=.95, solid_capstyle='round')

    def _region_at(self, t):
        for region in self._wave_regions:
            if region['start'] <= t <= region['end']:
                return region
        return None

    def _hide_wave_tooltip(self, event=None):
        tip = self._wave_tooltip
        if tip is not None:
            try:
                tip.withdraw()
            except tk.TclError:
                pass

    def _show_wave_tooltip(self, event):
        if self._dragging:
            self._hide_wave_tooltip()
            return
        if event.inaxes != self._ax_wave or event.xdata is None:
            self._hide_wave_tooltip()
            return
        region = self._region_at(float(event.xdata))
        if region is None:
            self._hide_wave_tooltip()
            return

        verdict_text, color = _LOCAL_VERDICTS[region['verdict']]
        if self._wave_tooltip is None:
            tip = tk.Toplevel(self.root)
            tip.withdraw()
            tip.overrideredirect(True)
            try: tip.attributes('-topmost', True)
            except tk.TclError: pass
            tip.configure(bg='#343440')
            body = tk.Frame(tip, bg='#11111B', padx=12, pady=9)
            body.pack(padx=1, pady=1)
            self._tip_title = tk.Label(body, bg='#11111B', font=('Segoe UI', 10, 'bold'))
            self._tip_title.pack(anchor='w')
            self._tip_values = tk.Label(body, bg='#11111B', justify='left',
                                        font=('Consolas', 9))
            self._tip_values.pack(anchor='w', pady=(5, 4))
            self._tip_verdict = tk.Label(body, bg='#11111B',
                                         font=('Segoe UI', 11, 'bold'))
            self._tip_verdict.pack(anchor='w')
            self._wave_tooltip = tip

        value_lines = []
        for _, key, value, contribution in region['evidence']:
            shown = f'{value:.6f}' if abs(value) < 1000 else f'{value:,.2f}'
            direction = ('AI evidence' if contribution > 0 else
                         'authentic evidence' if contribution < 0 else 'neutral')
            value_lines.append(
                f'{self._display_feature_name(key)}: {shown}  ({direction})')
        if not value_lines:
            value_lines = ['No local feature evidence available']

        self._tip_title.configure(text=f"{region['name']} evidence", fg=color)
        self._tip_values.configure(text='\n'.join(value_lines), fg=color)
        self._tip_verdict.configure(text=verdict_text, fg=color)
        tip = self._wave_tooltip
        try:
            pointer_x = self.root.winfo_pointerx()
            pointer_y = self.root.winfo_pointery()
            tip.geometry(f'+{pointer_x + 16}+{pointer_y + 18}')
            tip.deiconify()
            tip.lift()
        except tk.TclError:
            pass

    # ── History save ───────────────────────────────────────────────────────────
    def _save_current_to_history(self):
        # Duplicates ARE allowed — the same song can appear multiple times at
        # different points in history. We guard only against this same instance
        # saving twice (e.g. on launch AND on close). Tracks loaded FROM history
        # are flagged so they don't get re-added.
        if not self.caps['history']:
            return
        if self._from_history:
            return
        if self._already_saved:
            return
        try:
            self.history.add_track(
                self.audio_path, self.ai_score, self.features,
                self.y, self.sr, self.D_db, self.wf_t, self.wf_y,
                y_audio=self.y_audio, sr_audio=self.sr_audio,
                feature_contributions=self.feature_contributions)
            self._already_saved = True
        except Exception as e:
            print(f'[history] save current error: {e}')

    def _on_close(self):
        if self._wave_tooltip is not None:
            try: self._wave_tooltip.destroy()
            except tk.TclError: pass
            self._wave_tooltip = None
        if self._engine:
            self._engine.close()
        self.root.destroy()

    # ── Layout skeleton ────────────────────────────────────────────────────────
    def _build_layout(self):
        # Top header (title + current plan + upgrade) — packed FIRST at the top
        # so it stays fixed above the scroll area.
        self._header = tk.Frame(self.root, bg=BG_MAIN)
        self._header.pack(side='top', fill='x')
        self._build_header()

        # Bottom bar (track analysis) — packed FIRST to anchor bottom
        self._bottom = tk.Frame(self.root, bg=BG_MAIN)
        self._bottom.pack(side='bottom', fill='x')
        if self.caps['dropdown']:
            self._build_dropdown()

        # Gauge section
        self._gauge_frame = tk.Frame(self.root, bg=BG_CARD)
        self._gauge_frame.pack(side='bottom', fill='x')
        self._build_gauge_section()

        # Playback controls bar (business tier) — packed to the bottom region
        # ABOVE the gauge but BELOW the scroll area, so Play/Restart always sit
        # directly under the waveform, never above the spectrogram.
        if self.caps['playback']:
            self._controls_bar = tk.Frame(self.root, bg=BG_MAIN)
            self._controls_bar.pack(side='bottom', fill='x')

        # Scrollable top area
        self._scroll = ctk.CTkScrollableFrame(
            self.root, fg_color=BG_MAIN,
            scrollbar_button_color='#1A2030',
            scrollbar_button_hover_color='#2A3040')
        self._scroll.pack(side='top', fill='both', expand=True)

        # Track name
        self._track_name_lbl = ctk.CTkLabel(
            self._scroll,
            text=os.path.basename(self.audio_path),
            font=ctk.CTkFont('Segoe UI', 20, 'bold'),
            text_color='#6A9AB0')
        self._track_name_lbl.pack(pady=(4, 6))

        if self.caps['spectrogram']:
            self._build_combined_plot()
        else:
            self._build_waveform_only()

        if self.caps['playback']:
            self._build_controls()

        # Lock every button until the plot has fully rendered — closes the window
        # where a click could kick off a second render and stack a duplicate
        # cover. The plot reveal calls _unlock_ui(); a fallback timer guarantees
        # the UI never stays locked if the reveal stalls.
        self._lock_ui()
        self.root.after(6000, self._unlock_ui)

        # Gauge animation fires at 500ms — by then everything is rendered.
        # Save to history right as the animation begins.
        def _animate_and_save():
            self.gauge.animate_to(self.ai_score)
            self._save_current_to_history()
        self.root.after(500, _animate_and_save)
        # NOTE: _cache_mpl_bg (which also places the playhead at 0.0) is called
        # by the _fit loop once the plot has drawn at its final size — no need
        # to schedule it here, where it would race the not-yet-drawn canvas.
        self.root.update_idletasks()

    # ── Button lock (active while the plot is still rendering) ─────────────────
    def _lock_ui(self):
        for b in self._ui_buttons:
            try: b.configure(state='disabled')
            except Exception: pass

    def _unlock_ui(self):
        if self._ui_unlocked:
            return
        self._ui_unlocked = True
        for b in self._ui_buttons:
            try: b.configure(state='normal')
            except Exception: pass

    # ── Header (title + current plan + upgrade button) ─────────────────────────
    def _build_header(self):
        h = self._header

        # App title (left)
        tk.Label(h, text='AI Music Detector', bg=BG_MAIN, fg='#6A9AB0',
                 font=('Segoe UI', 15, 'bold')).pack(side='left', padx=16, pady=8)

        # Plan + upgrade (right)
        right = tk.Frame(h, bg=BG_MAIN)
        right.pack(side='right', padx=14, pady=6)

        name, color = PLAN_DISPLAY.get(self.tier, PLAN_DISPLAY['basic_model'])

        if self.tier == 'enterprise_model':
            # Max tier — no upgrade path. The Enterprise label sits where the
            # Upgrade button would normally be.
            tk.Label(right, text='Enterprise', bg=BG_MAIN, fg=color,
                     font=('Segoe UI', 14, 'bold')).pack(side='right')
            return

        # Upgrade button (rightmost). Gray box + lighter-gray text by default;
        # glows blue with white text on hover.
        btn = tk.Button(right, text='Change Plan', command=self._open_upgrade,
                        bg=UPGRADE_BG, fg=UPGRADE_FG,
                        activebackground=UPGRADE_BG_H, activeforeground=UPGRADE_FG_H,
                        relief='flat', bd=0, font=('Segoe UI', 12, 'bold'),
                        cursor='hand2', padx=18, pady=7)
        btn.pack(side='right', padx=(12, 0))
        btn.bind('<Enter>', lambda e: btn.config(bg=UPGRADE_BG_H, fg=UPGRADE_FG_H))
        btn.bind('<Leave>', lambda e: btn.config(bg=UPGRADE_BG,   fg=UPGRADE_FG))
        self._ui_buttons.append(btn)

        # Current plan label, to the LEFT of the button.
        tk.Label(right, text=f'{name} Plan', bg=BG_MAIN, fg=color,
                 font=('Segoe UI', 13, 'bold')).pack(side='right')

    # ── Upgrade page (shown in-place, over the main GUI) ───────────────────────
    def _open_upgrade(self):
        if self._upgrade_overlay is not None:
            return
        # Pause playback so audio doesn't keep going behind the upgrade page.
        if self._engine and self._playing:
            self._pause()

        # A borderless child window stays above the detector while its widgets
        # are rebuilt. A Frame inside root can briefly fall behind newly-created
        # Matplotlib/Tk widgets, which exposed the main screen mid-plan-change.
        self.root.update_idletasks()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        overlay = tk.Toplevel(self.root, bg='#040414')
        overlay.overrideredirect(True)
        overlay.transient(self.root)
        overlay.geometry(f'{root_w}x{root_h}+{root_x}+{root_y}')
        overlay.lift()
        try: overlay.grab_set()
        except tk.TclError: pass
        self._upgrade_overlay = overlay

        # Measure the overlay itself (the true visible viewport) AFTER it's laid
        # out — sizing off root before the overlay existed could mismatch the
        # real client area and push the Business card / Enterprise banner
        # off-screen.
        overlay.update_idletasks()
        sw = overlay.winfo_width()
        sh = overlay.winfo_height()
        if sw < 100: sw = self.root.winfo_width()
        if sh < 100: sh = self.root.winfo_height()

        # Tell the upgrade page which plan is currently active so it can mark it.
        name, _c = PLAN_DISPLAY.get(self.tier, PLAN_DISPLAY['basic_model'])

        # Import lazily to avoid a hard dependency at module load.
        from upgrade_plan import UpgradePlanView
        UpgradePlanView(overlay, sw, sh, on_back=self._close_upgrade,
                        current_plan=name,
                        on_plan_selected=self._change_plan,
                        current_user_email=self.current_user_email,
                        auth_token=self.auth_token)

    def _change_plan(self, plan_name):
        """Rebuild this analysis at another tier while the chooser blocks input."""
        if self._plan_change_in_progress:
            return
        tier_by_name = {
            'Basic': 'basic_model',
            'Premium': 'premium_model',
            'Business': 'business_model',
        }
        new_tier = tier_by_name.get(plan_name)
        if new_tier is None or new_tier == self.tier:
            return
        self._plan_change_in_progress = True

        global CURRENT_MODEL
        CURRENT_MODEL = new_tier
        # AnalysisRunner uses this value for future Upload New Track operations.
        try:
            import appGUI_Startup
            appGUI_Startup.CURRENT_TIER = new_tier
        except Exception as exc:
            print(f'[plan] could not synchronize startup tier: {exc}')

        root = self.root
        overlay = self._upgrade_overlay
        self._alive = False
        self._playing = False
        if self._engine:
            self._engine.close()

        # Preserve the blocking plan overlay while rebuilding the main GUI below.
        for widget in root.winfo_children():
            if widget is not overlay:
                widget.destroy()
        root.update_idletasks()

        new_app = AIDetectorApp(
            audio_path=self.audio_path,
            y=self.y, sr=self.sr, D_db=self.D_db,
            wf_t=self.wf_t, wf_y=self.wf_y,
            features=self.features, ai_score=self.ai_score,
            feature_contributions=self.feature_contributions,
            y_audio=self.y_audio, sr_audio=self.sr_audio,
            root=root, tier=new_tier, from_history=True,
            current_user_email=self.current_user_email,
            auth_token=self.auth_token)
        new_app._upgrade_overlay = overlay

        def _show_updated_chooser(attempt=0):
            if overlay is None or not overlay.winfo_exists():
                return
            overlay.lift()
            if new_app._ui_unlocked or attempt >= 160:
                for child in overlay.winfo_children():
                    child.destroy()
                sw = max(overlay.winfo_width(), root.winfo_width())
                sh = max(overlay.winfo_height(), root.winfo_height())
                from upgrade_plan import UpgradePlanView
                UpgradePlanView(
                    overlay, sw, sh, on_back=new_app._close_upgrade,
                    current_plan=plan_name,
                    on_plan_selected=new_app._change_plan,
                    current_user_email=new_app.current_user_email,
                    auth_token=new_app.auth_token)
                new_app._plan_change_in_progress = False
                return
            root.after(50, lambda: _show_updated_chooser(attempt + 1))

        root.after(50, _show_updated_chooser)

    def _close_upgrade(self):
        if self._upgrade_overlay is not None:
            try: self._upgrade_overlay.grab_release()
            except Exception: pass
            try: self._upgrade_overlay.destroy()
            except Exception: pass
            self._upgrade_overlay = None

    # ── How tall the plot can be without forcing the page to scroll ────────────
    def _available_plot_height(self):
        # Remaining vertical space in the window after the fixed top/bottom bars
        # and the in-scroll track-name label. Sizing the plot to this keeps the
        # whole spectrogram + waveform (axes included) on-screen, no scrolling.
        self.root.update_idletasks()
        total = self.root.winfo_height()
        used = 0
        for w in (getattr(self, '_header', None),
                  getattr(self, '_gauge_frame', None),
                  getattr(self, '_controls_bar', None),
                  getattr(self, '_bottom', None),
                  getattr(self, '_track_name_lbl', None)):
            if w is not None:
                try:
                    if w.winfo_ismapped():
                        used += w.winfo_height()
                except Exception:
                    pass
        return total - used - 28   # breathing room for paddings

    # ── Combined spectrogram + waveform (premium / business) ──────────────────
    def _build_combined_plot(self):
        sw, sh = self._sw, self._sh
        dur = self._duration
        fig_w = (sw - 32) / 100
        fig_h = (sh * 0.42) / 100

        fig = Figure(figsize=(fig_w, fig_h), dpi=100, facecolor='#000000')
        gs  = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.30)
        ax_s = fig.add_subplot(gs[0])
        ax_w = fig.add_subplot(gs[1], sharex=ax_s)

        fig.subplots_adjust(left=0.05, right=0.995, top=0.87, bottom=0.10)
        ax_s.set_xlim(0, dur); ax_w.set_xlim(0, dur)

        ax_s.set_facecolor('#000000')
        librosa.display.specshow(self.D_db, sr=self.sr, hop_length=256,
                                 x_axis='time', y_axis='hz', ax=ax_s, cmap='magma')
        ax_s.set_xlim(0, dur)
        ax_s.set_ylabel('Hz', color='white', fontsize=10, labelpad=8)
        ax_s.set_xlabel('')
        ax_s.tick_params(colors='white', labelsize=9)
        for sp in ax_s.spines.values(): sp.set_edgecolor('#333333')
        ax_s.grid(True, color='#333333', linewidth=0.35, alpha=0.45)
        ax_s.set_title('Spectrogram', color='#888888', fontsize=10, pad=4)
        ax_s.margins(x=0)
        ax_s.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: '' if v == 0 else f'{int(v)}'))

        ax_w.set_facecolor('#000000')
        if self.tier == 'business_model':
            self._plot_forensic_waveform(ax_w)
        else:
            ax_w.plot(self.wf_t, self.wf_y, color='#40E0D0',
                      linewidth=0.6, alpha=0.9)
        ax_w.set_ylabel('Amplitude', color='white', fontsize=10, labelpad=8)
        ax_w.set_xlabel('Time (s)',  color='white', fontsize=10)
        ax_w.tick_params(colors='white', labelsize=9)
        for sp in ax_w.spines.values(): sp.set_edgecolor('#333333')
        ax_w.axhline(0, color='#333333', linewidth=0.5)
        ax_w.grid(True, color='#333333', linewidth=0.35, alpha=0.45)
        ax_w.set_title('Waveform', color='#888888', fontsize=10, pad=4)
        ax_w.set_xlim(0, dur); ax_w.margins(x=0)

        target_ticks = 10
        raw_step = dur / target_ticks
        for step in (5, 10, 15, 30, 60, 120, 300, 600):
            if raw_step <= step:
                tick_step = step; break
        else:
            tick_step = 600
        ticks = np.arange(tick_step, dur, tick_step)
        ax_w.set_xticks(ticks)
        ax_w.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}'))

        # Playhead only in business tier
        if self.caps['playback']:
            self._vline_spec = ax_s.axvline(x=0, color='white', alpha=0.80,
                                            lw=1.5, zorder=10, animated=True)
            self._vline_wave = ax_w.axvline(x=0, color='white', alpha=0.80,
                                            lw=1.5, zorder=10, animated=True)
            trans = blended_transform_factory(ax_s.transData, ax_s.transAxes)
            self._triangle, = ax_s.plot(
                [0], [1.08], '>', color='#22FF66', markersize=16,
                markeredgecolor='white', markeredgewidth=1.5,
                transform=trans, clip_on=False, zorder=20, animated=True)

        self._ax_spec = ax_s
        self._ax_wave = ax_w
        self._fig     = fig

        canvas = FigureCanvasTkAgg(fig, master=self._scroll)
        tkw = canvas.get_tk_widget()
        locked_h = int(self._sh * 0.30)
        tkw.configure(bg='#000000', highlightthickness=0, height=locked_h)
        tkw.pack_propagate(False)
        # Pack the canvas immediately, but keep an OPAQUE cover placed on top of
        # it (a sibling inside the scroll, so it stays above the canvas through
        # matplotlib's resizes). Only drop the cover once the canvas has settled
        # at its final size and the playhead is in place — one clean reveal.
        tkw.pack(fill='x', padx=6, pady=(0, 2))

        # Cover the plot with a FIXED-height frame (the max the plot can grow to),
        # so it never resizes when the reveal changes the canvas height — a
        # resizing cover re-centers its label and leaves a ghost ("second"
        # label). Colour it like the loading overlay (#06060F) so if that
        # overlay's ring bleeds next to the cover, it's invisible instead of a
        # faint gray box.
        cover = tk.Frame(self._scroll, bg='#06060F')
        cover.place(in_=tkw, x=0, y=0, relwidth=1.0, height=int(self._sh * 0.5))
        cover.lift()
        tk.Label(cover, text='Rendering spectrogram + waveform…', bg='#06060F',
                 fg='#445566', font=('Segoe UI', 17)).place(
                     relx=0.5, rely=0.28, anchor='center')

        def _reveal(attempt=0, last_w=0, stable_count=0):
            # Wait for the REAL mapped canvas width to settle.
            w = tkw.winfo_width()
            if w == last_w and w > 100:
                stable_count += 1
            else:
                stable_count = 0
            if w > 100 and stable_count >= 3:
                # Shrink the plot to the remaining viewport so the waveform's
                # time axis is visible without scrolling.
                lh = max(200, min(self._available_plot_height(),
                                  int(self._sh * 0.5)))
                if abs(lh - tkw.winfo_height()) > 2:
                    tkw.configure(height=lh)
                    self.root.update_idletasks()
                # Draw once at the final size, then capture the blit background
                # from the actual mapped canvas (so it matches exactly — no
                # trails, no ghost plot) and stamp the playhead at the start.
                canvas.draw()
                if self.caps.get('playback'):
                    self._pos = 0.0
                    self._mpl_bg = canvas.copy_from_bbox(fig.bbox)
                    self._move_playhead(0.0)
                cover.lift()
                cover.destroy()    # reveal the finished plot in one shot
                self._unlock_ui()  # build finished — buttons are safe now
            elif attempt < 120:
                self.root.after(50, lambda: _reveal(attempt + 1, w, stable_count))
        self.root.after(50, _reveal)

        self._mpl_canvas = canvas
        self._fig_refs.append(fig)
        if self.tier == 'business_model':
            canvas.mpl_connect('motion_notify_event', self._show_wave_tooltip)
            canvas.mpl_connect('figure_leave_event', self._hide_wave_tooltip)
        if self.caps['playback']:
            canvas.mpl_connect('button_press_event',   self._on_press)
            canvas.mpl_connect('motion_notify_event',  self._on_motion)
            canvas.mpl_connect('button_release_event', self._on_release)
            # Any later canvas resize invalidates the cached blit background —
            # drop it so the next playhead move recaptures a fresh, matching one
            # (otherwise a stale background leaves trails / a ghost plot).
            canvas.mpl_connect('resize_event',
                               lambda e: setattr(self, '_mpl_bg', None))

    # ── Waveform only (basic tier — enlarged to fill spec+wave area) ──────────
    def _build_waveform_only(self):
        sw, sh = self._sw, self._sh
        dur = self._duration
        fig_w = (sw - 32) / 100
        fig_h = (sh * 0.55) / 100

        fig = Figure(figsize=(fig_w, fig_h), dpi=100, facecolor='#000000')
        ax_w = fig.add_subplot(111)
        fig.subplots_adjust(left=0.05, right=0.995, top=0.92, bottom=0.12)

        ax_w.set_facecolor('#000000')
        if self.tier == 'business_model':
            self._plot_forensic_waveform(ax_w)
        else:
            ax_w.plot(self.wf_t, self.wf_y, color='#40E0D0',
                      linewidth=0.6, alpha=0.9)
        ax_w.set_ylabel('Amplitude', color='white', fontsize=10, labelpad=8)
        ax_w.set_xlabel('Time (s)',  color='white', fontsize=10)
        ax_w.tick_params(colors='white', labelsize=9)
        for sp in ax_w.spines.values(): sp.set_edgecolor('#333333')
        ax_w.axhline(0, color='#333333', linewidth=0.5)
        ax_w.grid(True, color='#333333', linewidth=0.35, alpha=0.45)
        ax_w.set_title('Waveform', color='#888888', fontsize=10, pad=4)
        ax_w.set_xlim(0, dur); ax_w.margins(x=0)

        target_ticks = 10
        raw_step = dur / target_ticks
        for step in (5, 10, 15, 30, 60, 120, 300, 600):
            if raw_step <= step:
                tick_step = step; break
        else:
            tick_step = 600
        ticks = np.arange(tick_step, dur, tick_step)
        ax_w.set_xticks(ticks)
        ax_w.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}'))

        self._ax_wave = ax_w
        self._fig     = fig

        canvas = FigureCanvasTkAgg(fig, master=self._scroll)
        tkw = canvas.get_tk_widget()
        locked_h = int(self._sh * 0.36)
        tkw.configure(bg='#000000', highlightthickness=0, height=locked_h)
        tkw.pack_propagate(False)
        # Pack immediately, hidden under an opaque cover until the canvas settles.
        tkw.pack(fill='x', padx=6, pady=(0, 2))

        # Fixed-height cover coloured like the loading overlay (see the combined
        # plot for the reasoning: no resize ghost, no faint gray-box bleed).
        cover = tk.Frame(self._scroll, bg='#06060F')
        cover.place(in_=tkw, x=0, y=0, relwidth=1.0, height=int(self._sh * 0.5))
        cover.lift()
        tk.Label(cover, text='Rendering waveform…', bg='#06060F',
                 fg='#445566', font=('Segoe UI', 17)).place(
                     relx=0.5, rely=0.28, anchor='center')

        def _reveal(attempt=0, last_w=0, stable_count=0):
            w = tkw.winfo_width()
            if w == last_w and w > 100:
                stable_count += 1
            else:
                stable_count = 0
            if w > 100 and stable_count >= 3:
                lh = max(200, min(self._available_plot_height(),
                                  int(self._sh * 0.5)))
                if abs(lh - tkw.winfo_height()) > 2:
                    tkw.configure(height=lh)
                    self.root.update_idletasks()
                canvas.draw()
                cover.lift()
                cover.destroy()    # reveal the finished plot in one shot
                self._unlock_ui()  # build finished — buttons are safe now
            elif attempt < 120:
                self.root.after(50, lambda: _reveal(attempt + 1, w, stable_count))
        self.root.after(50, _reveal)

        self._mpl_canvas = canvas
        self._fig_refs.append(fig)
        if self.tier == 'business_model':
            canvas.mpl_connect('motion_notify_event', self._show_wave_tooltip)
            canvas.mpl_connect('figure_leave_event', self._hide_wave_tooltip)

    def _cache_mpl_bg(self):
        if self._mpl_canvas and self._fig and self.caps['playback']:
            # Draw the static plot, capture it as the blit background, then
            # stamp the playhead at 0.0 so it's visible immediately.
            self._mpl_canvas.draw()
            self._mpl_bg = self._mpl_canvas.copy_from_bbox(self._fig.bbox)
            self._move_playhead(0.0)

    # ── Controls (business only) ───────────────────────────────────────────────
    def _build_controls(self):
        ctrl = tk.Frame(self._controls_bar, bg=BG_MAIN)
        ctrl.pack(anchor='w', padx=8, pady=(4, 6))
        kw = dict(bg=BG_BTN, fg='#AABBCC', activebackground=BG_BTN_H,
                  activeforeground='white', relief='flat', bd=0,
                  font=FONT_BTN, cursor='hand2', padx=14, pady=6)
        self._play_btn = tk.Button(ctrl, text='▶  Play',
                                   command=self._toggle_play, **kw)
        self._play_btn.pack(side='left', padx=(0, 6))
        restart_btn = tk.Button(ctrl, text='⟳  Restart',
                  command=self._restart, **kw)
        restart_btn.pack(side='left', padx=(0, 6))
        self._ui_buttons.append(self._play_btn)
        self._ui_buttons.append(restart_btn)
        self._time_lbl = tk.Label(ctrl, text='0:00 / 0:00',
                                  bg=BG_MAIN, fg='#3A4A5A',
                                  font=('Segoe UI', 10))
        self._time_lbl.pack(side='left', padx=(10, 0))

    # ── Gauge section + buttons under it ───────────────────────────────────────
    def _build_gauge_section(self):
        f = self._gauge_frame

        # Binary confidence is confidence in whichever class won—not P(AI)
        # alone. For example, P(AI)=23% means P(Human)=77%, hence 77%
        # confidence in the displayed classification.
        confidence_pct = int(round(
            max(float(self.ai_score), 1.0 - float(self.ai_score)) * 100))
        conf_col = confidence_color(confidence_pct)
        pct = int(round(float(self.ai_score) * 100))
        score_col = score_color(self.ai_score)
        classification, classification_col = classification_for_score(
            self.ai_score)

        # The gauge retains its original centred position. The interpretation
        # card is positioned independently at the far right, so adding the card
        # can never push the gauge or its heading sideways.
        gauge_col = tk.Frame(f, bg=BG_CARD)
        gauge_col.pack(anchor='center', pady=(4, 0))

        # Three adjacent labels allow only the number and percent sign to carry
        # the confidence color while the surrounding heading remains white.
        score_title = tk.Frame(gauge_col, bg=BG_CARD)
        score_title.pack(pady=(8, 0))
        title_font = ('Segoe UI', 22, 'bold')
        tk.Label(score_title, text='AI Detection Score (', bg=BG_CARD,
                 fg='#FFFFFF', font=title_font).pack(side='left')
        tk.Label(score_title, text=f'{confidence_pct}%', bg=BG_CARD,
                 fg=conf_col, font=title_font).pack(side='left')
        tk.Label(score_title, text=' Confidence)', bg=BG_CARD,
                 fg='#FFFFFF', font=title_font).pack(side='left')

        gauge_size = max(440, min(int(self._sh * 0.36), 640))
        self.gauge = GaugeMeter(gauge_col, size=gauge_size)
        self.gauge.pack(pady=(2, 0))

        ctk.CTkLabel(gauge_col, text=f'{pct}%',
                     font=ctk.CTkFont('Segoe UI', 56, 'bold'),
                     text_color=score_col, fg_color=BG_CARD).pack(pady=(0, 2))

        # ── Button row under the gauge ───────────────────────────────────────
        btn_row = tk.Frame(gauge_col, bg=BG_CARD)
        btn_row.pack(pady=(2, 10))

        # Plain-language score interpretation. Segmented labels allow the two
        # percentages and classification itself to carry their semantic colour
        # while the explanatory wording remains white.
        panel_width = max(500, min(740, int(self._sw * 0.34)))
        panel_height = 340
        interpretation = tk.Frame(
            f, bg='#1A1A24', width=panel_width, height=panel_height,
            highlightthickness=1, highlightbackground='#30303D')
        # Twelve pixels matches the plot's visual right-hand inset closely.
        interpretation.place(relx=1.0, x=-12, y=42, anchor='ne')
        interpretation.pack_propagate(False)

        panel_inner = tk.Frame(interpretation, bg='#1A1A24')
        panel_inner.pack(fill='both', expand=True, padx=26, pady=24)
        panel_font = title_font

        def styled_wrapping_line(parts, lines, top_pad=0):
            """Create word-wrapped text while retaining per-part colours."""
            text = tk.Text(
                panel_inner, height=lines, wrap='word', bg='#1A1A24',
                fg='#FFFFFF', font=panel_font, relief='flat', bd=0,
                highlightthickness=0, padx=0, pady=0, cursor='arrow',
                takefocus=False)
            text.pack(fill='x', anchor='w', pady=(top_pad, 0))
            text.tag_configure('white', foreground='#FFFFFF')
            for index, (value, colour) in enumerate(parts):
                tag = f'colour_{index}'
                text.tag_configure(tag, foreground=colour)
                text.insert('end', value, 'white' if colour == '#FFFFFF' else tag)
            text.configure(state='disabled')
            return text

        styled_wrapping_line([
            ('AI Likelihood: ', '#FFFFFF'), (f'{pct}%', score_col)
        ], lines=1)

        classification_lines = (1 if len(classification) <= 28 else
                                2 if len(classification) <= 55 else 3)
        styled_wrapping_line([
            ('Classification: ', '#FFFFFF'),
            (classification, classification_col)
        ], lines=classification_lines, top_pad=28)

        styled_wrapping_line([
            ('Model Confidence in Classification: ', '#FFFFFF'),
            (f'{confidence_pct}%', conf_col)
        ], lines=2, top_pad=28)

        # # Choose Background (business only) — left of History
        # if self.caps['background']:
        #     tk.Button(btn_row, text='🖼  Choose Background',
        #               command=self._choose_background,
        #               bg=BG_BTN, fg='#AABBCC',
        #               activebackground=BG_BTN_H, activeforeground='white',
        #               relief='flat', bd=0, font=('Segoe UI', 11),
        #               cursor='hand2', padx=16, pady=8).pack(side='left', padx=6)

        # Upload New Track (all tiers)
        upload_btn = tk.Button(btn_row, text='Upload New Track',
                  command=self._upload_new_track,
                  bg='#1A1A22', fg='#FFFFFF',
                  activebackground='#26262E', activeforeground='white',
                  relief='flat', bd=0, font=('Segoe UI', 12, 'bold'),
                  cursor='hand2', padx=22, pady=9)
        upload_btn.pack(side='left', padx=6)
        self._ui_buttons.append(upload_btn)

        # History (premium + business)
        if self.caps['history']:
            history_btn = tk.Button(btn_row, text='↻  History',
                      command=self._toggle_history,
                      bg=BG_BTN, fg='#AABBCC',
                      activebackground=BG_BTN_H, activeforeground='white',
                      relief='flat', bd=0, font=('Segoe UI', 11),
                      cursor='hand2', padx=16, pady=8)
            history_btn.pack(side='left', padx=6)
            self._ui_buttons.append(history_btn)

    # ── Track analysis dropdown ────────────────────────────────────────────────
    def _build_dropdown(self):
        b = self._bottom
        # tk.Button (not CTkButton) so hover reliably drives BOTH the box colour
        # and the text colour, exactly like the Upgrade button.
        self._dd_btn = tk.Button(
            b, text='▼   Track Analysis   —   click to expand',
            command=self._toggle_dropdown,
            bg='#141420', fg='#555566',
            activebackground='#38BDF8', activeforeground='#FFFFFF',
            relief='flat', bd=0, font=('Segoe UI', 12),
            cursor='hand2', pady=9,
            highlightthickness=1, highlightbackground='#222232')
        self._dd_btn.pack(fill='x')
        # Hover: box glows light blue, text turns white; both revert on leave.
        self._dd_btn.bind('<Enter>',
                          lambda e: self._dd_btn.config(bg='#38BDF8', fg='#FFFFFF'))
        self._dd_btn.bind('<Leave>',
                          lambda e: self._dd_btn.config(bg='#141420', fg='#555566'))
        self._ui_buttons.append(self._dd_btn)

        self._dd_panel = ctk.CTkScrollableFrame(
            b, fg_color='#0C0C1A', height=280,
            border_width=1, border_color='#1E1E30')
        self._populate_dropdown()

    def _populate_dropdown(self):
        # Export button (business only) at the top of the panel
        if self.caps['export']:
            exp_row = tk.Frame(self._dd_panel, bg='#0C0C1A')
            exp_row.pack(fill='x', padx=16, pady=(8, 4))
            tk.Button(exp_row, text='Export 📄',
                      command=self._export,
                      bg=BG_BTN, fg='#AABBCC',
                      activebackground=BG_BTN_H, activeforeground='white',
                      relief='flat', bd=0, font=('Segoe UI', 11),
                      cursor='hand2', padx=14, pady=5).pack(side='left')

        for key, val in self.features.items():
            col = CMAP[classify(key, val)]
            row = tk.Frame(self._dd_panel, bg='#0C0C1A')
            row.pack(fill='x', padx=16, pady=1)
            tk.Label(row, text=key, bg='#0C0C1A', fg=col,
                     font=FONT_MONO, anchor='w').pack(side='left')
            vs = (f'{val:.6f}' if isinstance(val,float) and abs(val)<10000
                  else str(val))
            tk.Label(row, text=vs, bg='#0C0C1A', fg=col,
                     font=FONT_MONO, anchor='e').pack(side='right')

        leg = tk.Frame(self._dd_panel, bg='#0C0C1A')
        leg.pack(fill='x', padx=16, pady=(8, 12))
        for lbl, col in [('● Strongly human','#44CC44'),
                          ('● Uncertain','#CCCC22'),
                          ('● Strongly AI','#FF4444'),
                          ('● No threshold','#777777')]:
            tk.Label(leg, text=lbl, bg='#0C0C1A', fg=col,
                     font=('Segoe UI',9)).pack(side='left', padx=8)

        link = tk.Label(self._dd_panel, text='See Dictionary', bg='#0C0C1A',
                        fg='#5A8FC7', cursor='hand2',
                        font=('Segoe UI', 12, 'underline'))
        link.pack(anchor='w', padx=24, pady=(2, 14))
        link.bind('<Enter>', lambda _e: link.config(fg='#38BDF8'))
        link.bind('<Leave>', lambda _e: link.config(fg='#5A8FC7'))
        link.bind('<Button-1>', lambda _e: self._open_dictionary())

    def _open_dictionary(self):
        win = getattr(self, '_dictionary_window', None)
        if win is not None:
            try:
                win.win.lift(); win.win.focus_force(); return
            except tk.TclError:
                pass
        from feature_dictionary import FeatureDictionaryWindow
        self._dictionary_window = FeatureDictionaryWindow(
            self.root, list(self.features.keys()), _RULES,
            icon_setter=set_music_note_icon)

    def _toggle_dropdown(self):
        if self.dropdown_open:
            self._dd_panel.pack_forget()
            self._dd_btn.configure(text='▼   Track Analysis   —   click to expand')
        else:
            self._dd_panel.pack(fill='x', before=self._dd_btn)
            self._dd_btn.configure(text='▲   Track Analysis   —   click to collapse')
        self.dropdown_open = not self.dropdown_open

    # ── Export ───────────────────────────────────────────────────────────────
    def _export(self):
        # Reuse an already-open dialog rather than stacking a second one.
        win = getattr(self, '_export_win', None)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return

        win = tk.Toplevel(self.root)
        self._export_win = win
        win.title('Export As')
        win.configure(bg='#0C0C1A')
        # Replace Tk's default feather logo with a printer icon.
        try:
            win._icon_img = self._printer_icon()
            win.iconphoto(False, win._icon_img)
        except Exception:
            pass
        w, h = 360, 380
        x = (self._sw - w) // 2
        y = (self._sh - h) // 2
        win.geometry(f'{w}x{h}+{x}+{y}')
        win.minsize(300, 340)
        win.transient(self.root)

        tk.Label(win, text='Export formats', bg='#0C0C1A', fg='#FFFFFF',
                 font=('Segoe UI', 14, 'bold')).pack(anchor='w',
                                                     padx=20, pady=(18, 10))

        # Button row is packed BEFORE the body so its space is reserved at the
        # bottom — the buttons stay visible even at the minimum window size.
        btn_row = tk.Frame(win, bg='#0C0C1A')
        btn_row.pack(fill='x', side='bottom', padx=20, pady=(6, 14))
        exp_btn = tk.Button(btn_row, text='Export', command=self._do_export,
                            bg=BG_BTN, fg='#FFFFFF',
                            activebackground=BG_BTN_H, activeforeground='#FFFFFF',
                            relief='flat', bd=0, font=('Segoe UI', 11, 'bold'),
                            cursor='hand2', padx=18, pady=6)
        exp_btn.pack(side='right')
        cancel_btn = tk.Button(btn_row, text='Cancel', command=self._close_export,
                               bg=BG_BTN, fg='#AABBCC',
                               activebackground=BG_BTN_H, activeforeground='#FFFFFF',
                               relief='flat', bd=0, font=('Segoe UI', 11),
                               cursor='hand2', padx=18, pady=6)
        cancel_btn.pack(side='left')

        formats = ['.pdf', '.csv', '.txt', '.bin']
        self._export_sel   = {fmt: False for fmt in formats}
        self._export_boxes = {}

        body = tk.Frame(win, bg='#0C0C1A')
        body.pack(fill='both', expand=True, padx=20)

        for fmt in formats:
            row = tk.Frame(body, bg='#0C0C1A')
            row.pack(fill='x', pady=6)
            box = tk.Label(row, text=' ', width=2,
                           bg='#141428', fg='#44CC44',
                           font=('Segoe UI', 12, 'bold'),
                           relief='solid', bd=1)
            box.pack(side='left')
            lbl = tk.Label(row, text=fmt, bg='#0C0C1A', fg='#FFFFFF',
                           font=('Segoe UI', 12))
            lbl.pack(side='left', padx=10)
            self._export_boxes[fmt] = box
            for widget in (box, lbl):
                widget.configure(cursor='hand2')
                widget.bind('<Button-1>', lambda e, f=fmt: self._toggle_export(f))

    def _printer_icon(self):
        # A small printer icon used as the export window's title-bar / taskbar
        # icon: a sheet feeding in the top, the printer body, and a printed
        # sheet in the output tray.
        from PIL import Image, ImageDraw, ImageTk
        S = 64
        img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        outline = (120, 130, 150, 255)
        paper   = (245, 246, 250, 255)
        # Sheet feeding in from the top.
        d.rectangle([21, 8, 43, 24], fill=paper, outline=outline)
        # Printer body.
        d.rectangle([12, 22, 52, 44], fill=(96, 106, 126, 255), outline=outline)
        # Status LED.
        d.ellipse([45, 26, 49, 30], fill=(80, 210, 120, 255))
        # Printed sheet in the output tray, with text lines.
        d.rectangle([20, 39, 44, 56], fill=paper, outline=outline)
        for i, ly in enumerate(range(44, 54, 4)):
            lx2 = 40 if i % 2 == 0 else 34
            d.line([(24, ly), (lx2, ly)], fill=(150, 160, 175, 255), width=2)
        return ImageTk.PhotoImage(img)

    def _close_export(self):
        if getattr(self, '_export_win', None) is not None:
            self._export_win.destroy()
            self._export_win = None

    def _toggle_export(self, fmt):
        self._export_sel[fmt] = not self._export_sel[fmt]
        self._export_boxes[fmt].configure(
            text='✓' if self._export_sel[fmt] else ' ')

    def _do_export(self):
        selected = [f for f, v in self._export_sel.items() if v]
        if not selected:
            return
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        writers = {
            '.pdf': self._export_pdf,
            '.csv': self._export_csv,
            '.txt': self._export_txt,
            '.bin': self._export_bin,
        }
        for fmt in selected:
            directory = filedialog.askdirectory(
                parent=self._export_win,
                title=f'Choose a folder for the {fmt} export')
            if not directory:
                continue   # user cancelled this format's directory picker
            path = os.path.join(directory, base + fmt)
            try:
                writers[fmt](path)
            except Exception as e:
                print(f'[export] failed for {fmt}: {e}')
        self._export_win.destroy()
        self._export_win = None

    # ── Export writers ───────────────────────────────────────────────────────
    def _export_verdict(self):
        return ('Strongly AI'    if self.ai_score > 0.65 else
                'Uncertain'      if self.ai_score > 0.35 else
                'Strongly Human')

    def _feature_str(self, val):
        return (f'{val:.6f}' if isinstance(val, float) and abs(val) < 10000
                else str(val))

    def _export_txt(self, path):
        lines = [
            'AI Music Detection — Analysis Report',
            '=' * 40,
            f'File: {os.path.basename(self.audio_path)}',
            f'AI Detection Score: {self.ai_score * 100:.1f}%',
            f'Verdict: {self._export_verdict()}',
            '',
            'Features:',
        ]
        for k, v in self.features.items():
            lines.append(f'  {k}: {self._feature_str(v)}')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def _export_csv(self, path):
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', os.path.basename(self.audio_path)])
            w.writerow(['ai_detection_score', f'{self.ai_score:.6f}'])
            w.writerow(['verdict', self._export_verdict()])
            w.writerow([])
            w.writerow(['feature', 'value'])
            for k, v in self.features.items():
                w.writerow([k, v])

    def _export_bin(self, path):
        import pickle
        payload = {
            'file':     os.path.basename(self.audio_path),
            'ai_score': self.ai_score,
            'verdict':  self._export_verdict(),
            'features': self.features,
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    def _export_pdf(self, path):
        from matplotlib.backends.backend_pdf import PdfPages
        fig = Figure(figsize=(8.27, 11.69))   # A4 portrait
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        y = 0.96
        ax.text(0.08, y, 'AI Music Detection — Analysis Report',
                fontsize=18, fontweight='bold', va='top')
        y -= 0.05
        ax.text(0.08, y, f'File: {os.path.basename(self.audio_path)}',
                fontsize=11, va='top')
        y -= 0.03
        ax.text(0.08, y, f'AI Detection Score: {self.ai_score * 100:.1f}%',
                fontsize=11, va='top')
        y -= 0.03
        ax.text(0.08, y, f'Verdict: {self._export_verdict()}',
                fontsize=11, va='top')
        y -= 0.05
        ax.text(0.08, y, 'Features:', fontsize=13, fontweight='bold', va='top')
        y -= 0.035
        for k, v in self.features.items():
            ax.text(0.10, y, f'{k}: {self._feature_str(v)}',
                    fontsize=9, va='top', family='monospace')
            y -= 0.022
            if y < 0.04:
                break
        with PdfPages(path) as pdf:
            pdf.savefig(fig)

    # ══════════════════════════════════════════════════════════════════════════
    #  Upload New Track  → reuse the loading overlay from appGUI_Startup
    # ══════════════════════════════════════════════════════════════════════════
    def _upload_new_track(self):
        path = filedialog.askopenfilename(
            title='Select Audio Track',
            filetypes=[('Audio Files','*.mp3 *.wav *.flac *.ogg *.aac *.m4a'),
                       ('All Files','*.*')])
        if not path:
            return
        # Make sure current track is saved before we replace it
        self._save_current_to_history()
        # Stop playback loop + audio from THIS (soon-to-be-replaced) instance
        self._alive   = False
        self._playing = False
        if self._engine:
            self._engine.close()
        # Import here to avoid circular import at module load
        from appGUI_Startup import AnalysisRunner
        AnalysisRunner(self.root, path,
                       on_done=self._on_new_track_done,
                       on_cancelled=self._on_new_track_cancelled)

    def _on_new_track_cancelled(self):
        """User cancelled the new-track analysis — revive this instance."""
        self._alive = True   # we were never torn down; keep running

    def _on_new_track_done(self, result, overlay):
        """New track analyzed — rebuild the main GUI in place."""
        tier = self.tier
        root = self.root
        # Destroy everything except the overlay, then rebuild
        overlay_frame = overlay.frame if overlay else None
        for w in root.winfo_children():
            if w is not overlay_frame:
                w.destroy()
        root.update_idletasks()

        new_app = AIDetectorApp(
            audio_path=result['audio_path'],
            y=result['y'], sr=result['sr'], D_db=result['D_db'],
            wf_t=result['wf_t'], wf_y=result['wf_y'],
            features=result['features'], ai_score=result['ai_score'],
            feature_contributions=result.get('feature_contributions'),
            y_audio=result['y_audio'], sr_audio=result['sr_audio'],
            root=root, tier=tier,
            current_user_email=self.current_user_email,
            auth_token=self.auth_token)

        # Close the loading overlay shortly after the new canvas first draws; the
        # plot's own cover keeps hiding it until fully rendered.
        def _close_overlay(event=None):
            if not hasattr(_close_overlay, '_fired'):
                _close_overlay._fired = True
                if overlay:
                    root.after(400, overlay.close)
        if new_app._mpl_canvas:
            new_app._mpl_canvas.get_tk_widget().bind('<Map>', _close_overlay, add='+')
            new_app._mpl_canvas.mpl_connect('draw_event', _close_overlay)
        root.after(2500, _close_overlay)

    # ══════════════════════════════════════════════════════════════════════════
    #  Choose Background (business)
    # ══════════════════════════════════════════════════════════════════════════
    # def _choose_background(self):
    #     path = filedialog.askopenfilename(
    #         title='Select Background Image',
    #         filetypes=[('Image Files','*.png *.jpg *.jpeg *.bmp'),
    #                    ('All Files','*.*')])
    #     if not path:
    #         return
    #     SettingsStore.set('background_path', path)
    #     print(f'[background] saved: {path}')
    #     # Note: startup screen reads this on next launch.

    # ══════════════════════════════════════════════════════════════════════════
    #  History slide-out panel (premium + business)
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_history(self):
        if self._history_open:
            self._close_history()
        else:
            self._open_history()

    def _open_history(self):
        if self._history_open:
            return
        anim_serial = self._cancel_history_animation()
        self._history_open = True

        # A rapid reopen can otherwise leave the closing panel orphaned while a
        # second one is created. Always begin with exactly one drawer instance.
        if self._history_panel is not None:
            try: self._history_panel.destroy()
            except Exception: pass
            self._history_panel = None

        # Use the ACTUAL window size (not screen size — DPI-safe)
        self.root.update_idletasks()
        win_w = self.root.winfo_width()
        win_h = self.root.winfo_height()
        panel_w = int(win_w * 0.20)
        self._hist_win_w = win_w          # remember for close animation
        self._hist_win_h = win_h
        self._hist_panel_w = panel_w

        # Floating panel, starts fully off the right edge
        self._history_panel = tk.Frame(self.root, bg='#0A0A16',
                                        highlightbackground='#1E1E30',
                                        highlightthickness=1)
        self._history_panel.place(x=win_w, y=0, width=panel_w, height=win_h)
        self._history_panel.lift()        # float above main content

        # Header bar with X
        hdr = tk.Frame(self._history_panel, bg='#0A0A16')
        hdr.pack(fill='x', pady=(10, 4), padx=10)
        tk.Label(hdr, text='Track History', bg='#0A0A16', fg='#FFFFFF',
                 font=('Segoe UI', 15, 'bold')).pack(side='left')
        tk.Button(hdr, text='✕', command=self._close_history,
                  bg='#0A0A16', fg='#777788',
                  activebackground='#0A0A16', activeforeground='#FFFFFF',
                  relief='flat', bd=0, font=('Segoe UI', 14, 'bold'),
                  cursor='hand2').pack(side='right')

        # Scrollable list
        listwrap = ctk.CTkScrollableFrame(self._history_panel,
                                          fg_color='#0A0A16',
                                          scrollbar_button_color='#1A2030',
                                          scrollbar_button_hover_color='#2A3040')
        listwrap.pack(fill='both', expand=True, padx=6, pady=6)
        self._populate_history(listwrap)

        # Click-outside binding is enabled only AFTER slide-in finishes
        self._hist_click_bind = None

        self._animate_history(opening=True, anim_serial=anim_serial)

    def _cancel_history_animation(self):
        self._history_anim_serial += 1
        if self._history_anim_job is not None:
            try: self.root.after_cancel(self._history_anim_job)
            except Exception: pass
            self._history_anim_job = None
        return self._history_anim_serial

    def _enable_history_click_outside(self):
        if self._history_open and self._hist_click_bind is None:
            self._hist_click_bind = self.root.bind(
                '<Button-1>', self._history_click_outside, add='+')

    def _populate_history(self, parent):
        groups = self.history.grouped_by_date()
        if not groups:
            tk.Label(parent, text='No tracks analyzed yet.',
                     bg='#0A0A16', fg='#555566',
                     font=('Segoe UI', 11)).pack(pady=20)
            return
        for date_label, entries in groups:
            tk.Label(parent, text=date_label, bg='#0A0A16', fg='#6A9AB0',
                     font=('Segoe UI', 12, 'bold'), anchor='w').pack(
                         fill='x', padx=8, pady=(12, 4))
            for entry in entries:
                self._history_row(parent, entry)

    def _history_row(self, parent, entry):
        score_pct = int(round(entry['ai_score'] * 100))
        col = history_score_color(score_pct)

        row = tk.Frame(parent, bg='#0C0C1A',
                       highlightbackground='#0C0C1A', highlightthickness=1,
                       cursor='hand2')
        row.pack(fill='x', padx=8, pady=2)

        name = entry['filename']
        if len(name) > 26:
            name = name[:23] + '...'
        name_lbl = tk.Label(row, text=name, bg='#0C0C1A', fg='#CCCCDD',
                            font=('Segoe UI', 10), anchor='w')
        name_lbl.pack(side='left', padx=(8, 4), pady=6)

        score_lbl = tk.Label(row, text=f'{score_pct}%', bg='#0C0C1A', fg=col,
                             font=('Segoe UI', 11, 'bold'), anchor='e')
        score_lbl.pack(side='right', padx=(4, 8))

        # Hover: white border + brighter name. NOTE: do NOT change font sizes
        # here — resizing the row shifts its bottom edge under the cursor, which
        # makes <Enter>/<Leave> fire in a rapid loop (the flicker). Colour-only
        # feedback keeps the row height fixed, so no oscillation.
        def on_enter(e):
            row.configure(highlightbackground='#FFFFFF')
            name_lbl.configure(fg='#FFFFFF')
        def on_leave(e):
            row.configure(highlightbackground='#0C0C1A')
            name_lbl.configure(fg='#CCCCDD')
        for w in (row, name_lbl, score_lbl):
            w.bind('<Enter>', on_enter)
            w.bind('<Leave>', on_leave)
            w.bind('<Button-1>', lambda e, eid=entry['id']: self._load_history_track(eid))

    def _load_history_track(self, entry_id):
        data = self.history.load_track(entry_id)
        if not data:
            print('[history] could not load track')
            return
        # Tear down history panel synchronously (no slide animation race)
        self._history_open = False
        if self._hist_click_bind is not None:
            try: self.root.unbind('<Button-1>', self._hist_click_bind)
            except Exception: pass
            self._hist_click_bind = None
        if self._history_panel is not None:
            try: self._history_panel.destroy()
            except Exception: pass
            self._history_panel = None

        root = self.root
        tier = self.tier
        self._alive   = False
        self._playing = False
        if self._engine:
            self._engine.close()
        for w in root.winfo_children():
            w.destroy()
        root.update_idletasks()
        # Rebuild main GUI from cached data (no recompute).
        # from_history=True → this load will NOT be re-written to history.
        AIDetectorApp(
            audio_path=data['audio_path'],
            y=data['y'], sr=data['sr'], D_db=data['D_db'],
            wf_t=data['wf_t'], wf_y=data['wf_y'],
            features=data['features'], ai_score=data['ai_score'],
            feature_contributions=data.get('feature_contributions'),
            y_audio=data['y_audio'], sr_audio=data['sr_audio'],
            root=root, tier=tier, from_history=True,
            current_user_email=self.current_user_email,
            auth_token=self.auth_token)

    def _history_click_outside(self, event):
        if not self._history_open or not self._history_panel:
            return
        # Walk up the widget tree from whatever was clicked — if the panel
        # is an ancestor, the click was inside it; otherwise close.
        w = event.widget
        while w is not None:
            if w is self._history_panel:
                return   # click landed inside the panel — keep open
            try:
                w = w.master
            except Exception:
                break
        self._close_history()

    def _close_history(self):
        if not self._history_open:
            return
        self._history_open = False
        anim_serial = self._cancel_history_animation()
        if getattr(self, '_hist_click_bind', None) is not None:
            try: self.root.unbind('<Button-1>', self._hist_click_bind)
            except Exception: pass
            self._hist_click_bind = None
        self._animate_history(opening=False, anim_serial=anim_serial)

    def _animate_history(self, opening, anim_serial, step=0, start_x=None):
        if (not self._history_panel or
                anim_serial != self._history_anim_serial):
            return
        panel_w = getattr(self, '_hist_panel_w', int(self.root.winfo_width()*0.20))
        win_h = self.root.winfo_height()
        win_w = self.root.winfo_width()

        if start_x is None:
            # Continue from the drawer's actual current location. This makes a
            # close requested during slide-in reverse smoothly from that point.
            start_x = self._history_panel.winfo_x()

        # Fixed 12-step animation — guaranteed to terminate (no asymptote)
        TOTAL = 12
        if opening:
            end_x = win_w - panel_w
        else:
            # Travel one full panel width beyond the edge. Ending at win_w left
            # a thin strip visible during the final eased animation frames.
            end_x = win_w + panel_w

        if step >= TOTAL:
            self._history_anim_job = None
            if not opening:
                # Fully closed — destroy so nothing lingers on screen
                self._history_panel.place(x=end_x, y=0,
                                          width=panel_w, height=win_h)
                try: self._history_panel.destroy()
                except Exception: pass
                self._history_panel = None
            else:
                self._history_panel.place(x=end_x, y=0,
                                          width=panel_w, height=win_h)
                self._history_panel.lift()
                self._enable_history_click_outside()
            return

        # Ease-out interpolation
        frac = 1 - (1 - (step / TOTAL)) ** 2
        cur_x = int(start_x + (end_x - start_x) * frac)
        self._history_panel.place(x=cur_x, y=0, width=panel_w, height=win_h)
        self._history_panel.lift()
        self._history_anim_job = self.root.after(
            12, lambda: self._animate_history(
                opening, anim_serial, step + 1, start_x))

    # ══════════════════════════════════════════════════════════════════════════
    #  Playback (business)
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_play(self, event=None):
        if self._playing: self._pause()
        else:             self._play()

    def _on_space(self, event):
        self._toggle_play(); return 'break'

    def _play(self):
        self._engine.play(from_sec=self._pos)
        self._playing = True
        self._play_btn.configure(text='⏸  Pause')
        self._tick()

    def _pause(self):
        self._pos = self._engine.position
        self._engine.pause()
        self._playing = False
        self._play_btn.configure(text='▶  Play')

    def _restart(self):
        self._engine.stop()
        self._playing = False
        self._pos = 0.0
        self._play_btn.configure(text='▶  Play')
        self._move_playhead(0.0)
        self._fmt_time(0.0)

    def _on_track_end(self):
        if not self._alive: return
        self._playing = False
        self._pos = 0.0
        self._play_btn.configure(text='▶  Play')
        self._move_playhead(0.0)
        self._fmt_time(0.0)

    def _tick(self):
        if not self._alive or not self._playing: return
        t = self._engine.position
        self._move_playhead(t)
        self._fmt_time(t)
        self.root.after(50, self._tick)

    def _move_playhead(self, t):
        if not self._alive: return
        if None in (self._vline_spec, self._mpl_canvas):
            return
        try:
            # If we don't have a clean blit background yet, capture one now.
            # Without this, every move falls back to a full draw_idle() (slow),
            # which is what makes dragging laggy and the playhead invisible at
            # startup. The background excludes the playhead because the playhead
            # artists are animated=True (draw() skips them).
            if self._mpl_bg is None:
                self._mpl_canvas.draw()
                self._mpl_bg = self._mpl_canvas.copy_from_bbox(self._fig.bbox)
            self._vline_spec.set_xdata([t, t])
            self._vline_wave.set_xdata([t, t])
            self._triangle.set_xdata([t])
            c = self._mpl_canvas
            c.restore_region(self._mpl_bg)
            self._ax_spec.draw_artist(self._vline_spec)
            self._ax_wave.draw_artist(self._vline_wave)
            self._ax_spec.draw_artist(self._triangle)
            c.blit(self._fig.bbox)
        except Exception as e:
            print(f'[playhead] {e}')

    def _fmt_time(self, t):
        def s2m(s): return f'{int(s)//60}:{int(s)%60:02d}'
        self._time_lbl.config(text=f'{s2m(t)} / {s2m(self._duration)}')

    def _xdata(self, event):
        for ax in (self._ax_spec, self._ax_wave):
            if event.inaxes == ax and event.xdata is not None:
                return float(event.xdata)
        if self._ax_spec:
            try:
                xd, _ = self._ax_spec.transData.inverted().transform(
                            (event.x, event.y))
                ya = self._ax_spec.transAxes.inverted().transform(
                            (event.x, event.y))[1]
                if 0.88 <= ya <= 1.20:
                    return float(xd)
            except Exception:
                pass
        return None

    def _on_press(self, event):
        x = self._xdata(event)
        if x is None: return
        x = max(0.0, min(x, self._duration))
        self._dragging = True
        if self._playing:
            self._pos = self._engine.position
            self._engine.pause()
            self._playing = False
            self._play_btn.configure(text='▶  Play')
        # Ensure a clean blit background exists before dragging (so the old
        # playhead line gets wiped each frame instead of stacking). Only draw
        # if we don't already have one — avoids a full redraw on every click.
        if self._mpl_bg is None and self._mpl_canvas and self._fig:
            self._mpl_canvas.draw()
            self._mpl_bg = self._mpl_canvas.copy_from_bbox(self._fig.bbox)
        self._pos = x
        self._engine._frame = int(x * self._engine._sr)
        self._move_playhead(x)
        self._fmt_time(x)

    def _on_motion(self, event):
        if not self._dragging: return
        x = self._xdata(event)
        if x is None:
            if self._ax_spec:
                try:
                    x, _ = self._ax_spec.transData.inverted().transform(
                               (event.x, event.y))
                except Exception:
                    return
        if x is None: return
        x = max(0.0, min(float(x), self._duration))
        self._pos = x
        self._engine._frame = int(x * self._engine._sr)
        self._move_playhead(x)
        self._fmt_time(x)

    def _on_release(self, event):
        self._dragging = False

    def run(self):
        if self._owns_root:
            self.root.mainloop()
