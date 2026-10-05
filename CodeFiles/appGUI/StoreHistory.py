import os
import json
import datetime
import numpy as np
import sys

# ══════════════════════════════════════════════════════════════════════════════
#  HistoryStore
#  ─────────────────────────────────────────────────────────────────────────────
#  Persists analyzed-track history across program runs.
#
#  Storage layout (all next to the script files, in CodeFiles/):
#    history.json          — small metadata list (filename, date, score, id)
#    history_cache/<id>.npz — heavy display arrays for one track
#
#  To RESET history: delete history.json AND the history_cache/ folder.
#  (AIDetectorApp also exposes a reset_history() classmethod for convenience.)
# ══════════════════════════════════════════════════════════════════════════════

if getattr(sys, 'frozen', False):
    # Frozen applications may be installed in a read-only location, and a
    # one-file build is extracted to a temporary directory. Keep user-created
    # history and settings in a stable, writable Windows data directory.
    _BASE_DIR = os.path.join(
        os.environ.get('LOCALAPPDATA', os.path.expanduser('~')),
        'AI Music Detector')
    os.makedirs(_BASE_DIR, exist_ok=True)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH  = os.path.join(_BASE_DIR, 'history.json')
_CACHE_DIR  = os.path.join(_BASE_DIR, 'history_cache')


class HistoryStore:
    def __init__(self):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        self._entries = self._load_index()

    # ── Index (JSON) ───────────────────────────────────────────────────────────
    def _load_index(self):
        if not os.path.exists(_JSON_PATH):
            return []
        try:
            with open(_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_index(self):
        try:
            with open(_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._entries, f, indent=2)
        except Exception as e:
            print(f'[history] save error: {e}')

    # ── Public API ───────────────────────────────────────────────────────────
    def has_track(self, audio_path: str) -> bool:
        """True if a track with this absolute path is already in history."""
        ap = os.path.abspath(audio_path)
        return any(e.get('audio_path') == ap for e in self._entries)

    def add_track(self, audio_path, ai_score, features,
                  y, sr, D_db, wf_t, wf_y, y_audio=None, sr_audio=None,
                  feature_contributions=None):
        """
        Add a track to history. Duplicates ARE allowed — the same audio file
        may appear multiple times (each analysis is its own entry).
        Heavy arrays go into an .npz cache file; metadata into JSON.
        Returns the entry id.
        """
        ap = os.path.abspath(audio_path)

        # Unique id: timestamp (to the microsecond) + short hash, so two
        # analyses of the SAME file still get distinct ids and cache files.
        ts = datetime.datetime.now()
        entry_id = ts.strftime('%Y%m%d_%H%M%S_%f_') + str(abs(hash(ap)) % 10000)

        # Save heavy arrays to npz
        npz_path = os.path.join(_CACHE_DIR, f'{entry_id}.npz')
        try:
            save_kwargs = dict(
                y=y, D_db=D_db, wf_t=wf_t, wf_y=wf_y,
                sr=np.array([sr]),
                ai_score=np.array([ai_score]),
            )
            if y_audio is not None:
                save_kwargs['y_audio']  = y_audio
                save_kwargs['sr_audio'] = np.array([sr_audio
                                                    if sr_audio else sr])
            np.savez_compressed(npz_path, **save_kwargs)
            # Features stored as JSON-friendly dict inside a sidecar key file
            feat_path = os.path.join(_CACHE_DIR, f'{entry_id}_feat.json')
            with open(feat_path, 'w', encoding='utf-8') as f:
                json.dump(features, f)
            contrib_path = os.path.join(_CACHE_DIR, f'{entry_id}_contrib.json')
            with open(contrib_path, 'w', encoding='utf-8') as f:
                json.dump(feature_contributions or {}, f)
        except Exception as e:
            print(f'[history] cache write error: {e}')
            return None

        entry = {
            'id':         entry_id,
            'audio_path': ap,
            'filename':   os.path.basename(ap),
            'ai_score':   float(ai_score),
            'date':       ts.strftime('%Y-%m-%d'),
            'date_label': ts.strftime('%b %d, %Y'),   # e.g. "Jun 12, 2026"
            'time':       ts.strftime('%H:%M'),
            'timestamp':  ts.isoformat(),
        }
        # Newest first
        self._entries.insert(0, entry)
        self._save_index()
        return entry_id

    def all_entries(self):
        """Return entries newest-first."""
        return list(self._entries)

    def grouped_by_date(self):
        """
        Return an ordered list of (date_label, [entries]) groups,
        newest day first, entries within a day newest first.
        """
        groups = []
        seen   = {}
        for e in self._entries:                 # already newest-first
            lbl = e.get('date_label', e.get('date', 'Unknown'))
            if lbl not in seen:
                seen[lbl] = []
                groups.append((lbl, seen[lbl]))
            seen[lbl].append(e)
        return groups

    def load_track(self, entry_id):
        """
        Load full display data for a history entry.
        Returns dict with y, sr, D_db, wf_t, wf_y, features, ai_score,
        y_audio, sr_audio, audio_path  — or None on failure.
        """
        npz_path  = os.path.join(_CACHE_DIR, f'{entry_id}.npz')
        feat_path = os.path.join(_CACHE_DIR, f'{entry_id}_feat.json')
        contrib_path = os.path.join(_CACHE_DIR, f'{entry_id}_contrib.json')
        if not os.path.exists(npz_path):
            return None
        try:
            data = np.load(npz_path, allow_pickle=False)
            features = {}
            if os.path.exists(feat_path):
                with open(feat_path, 'r', encoding='utf-8') as f:
                    features = json.load(f)
            feature_contributions = {}
            if os.path.exists(contrib_path):
                with open(contrib_path, 'r', encoding='utf-8') as f:
                    feature_contributions = json.load(f)

            entry = next((e for e in self._entries if e['id'] == entry_id), {})

            result = {
                'y':        data['y'],
                'sr':       int(data['sr'][0]),
                'D_db':     data['D_db'],
                'wf_t':     data['wf_t'],
                'wf_y':     data['wf_y'],
                'features': features,
                'feature_contributions': feature_contributions,
                'ai_score': float(data['ai_score'][0]),
                'audio_path': entry.get('audio_path', ''),
                'y_audio':  data['y_audio']  if 'y_audio'  in data else None,
                'sr_audio': int(data['sr_audio'][0]) if 'sr_audio' in data else None,
            }
            return result
        except Exception as e:
            print(f'[history] load error: {e}')
            return None

    # ── Reset ────────────────────────────────────────────────────────────────
    @staticmethod
    def reset():
        """Delete all history (JSON + cache). Call to wipe everything."""
        try:
            if os.path.exists(_JSON_PATH):
                os.remove(_JSON_PATH)
            if os.path.isdir(_CACHE_DIR):
                for fn in os.listdir(_CACHE_DIR):
                    try: os.remove(os.path.join(_CACHE_DIR, fn))
                    except Exception: pass
            print('[history] reset complete')
        except Exception as e:
            print(f'[history] reset error: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  Settings store (for custom background — business tier)
# ══════════════════════════════════════════════════════════════════════════════
_SETTINGS_PATH = os.path.join(_BASE_DIR, 'app_settings.json')


class SettingsStore:
    @staticmethod
    def load():
        if not os.path.exists(_SETTINGS_PATH):
            return {}
        try:
            with open(_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def save(settings: dict):
        try:
            with open(_SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f'[settings] save error: {e}')

    @staticmethod
    def get(key, default=None):
        return SettingsStore.load().get(key, default)

    @staticmethod
    def set(key, value):
        s = SettingsStore.load()
        s[key] = value
        SettingsStore.save(s)
