import os


CODEFILES_DIR = os.path.abspath(os.path.join(SPECPATH, '..'))
PROJECT_DIR = os.path.dirname(CODEFILES_DIR)
APPGUI_DIR = os.path.join(CODEFILES_DIR, 'appGUI')

datas = []
binaries = []
hiddenimports = [
    'appGUI_Startup', 'appGUI_Main', 'upgrade_plan',
    'feature_dictionary', 'StoreHistory', 'computeinfo',
    'matplotlib.backends.backend_pdf', 'lightgbm', 'lightgbm.sklearn',
    'soundfile', 'sounddevice',
]

required_data = (
    (os.path.join(PROJECT_DIR, 'model_v2.pkl'), '.'),
    (os.path.join(PROJECT_DIR, 'MovingSongs'), 'MovingSongs'),
    (os.path.join(PROJECT_DIR, 'Screenshots_AIMusic'), 'Screenshots_AIMusic'),
)
for source, destination in required_data:
    if not os.path.exists(source):
        raise FileNotFoundError(f'Required application resource is missing: {source}')
    datas.append((source, destination))

a = Analysis(
    [os.path.join(APPGUI_DIR, 'LoginScreen.py')],
    pathex=[APPGUI_DIR, CODEFILES_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # These optional ecosystems are present but broken/incomplete in the global
    # development environment. The detector does not use them; excluding them
    # prevents scikit-learn's compatibility layer from pulling them into builds.
    excludes=[
        'torch', 'transformers', 'tensorflow', 'cupy', 'dask',
        'IPython', 'jupyter', 'zmq', 'pytest', 'openpyxl', 'fsspec',
        'pygame', 'yt_dlp',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI Music Detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI Music Detector',
)
