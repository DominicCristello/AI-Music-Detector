import os


CODEFILES_DIR = os.path.abspath(os.path.join(SPECPATH, '..'))
SERVER_DIR = os.path.join(CODEFILES_DIR, 'Server')

datas = []
binaries = []
hiddenimports = [
    'Server', 'Db', 'Auth', 'Config', 'Email_utils', 'Stripe_utils',
    'uvicorn.logging', 'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
]

a = Analysis(
    [os.path.join(SERVER_DIR, 'server_launcher.py')],
    pathex=[SERVER_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The auth server never imports the detector's audio/ML stack.  Keeping
    # these packages out makes the personal server build substantially lighter
    # and avoids loading hundreds of megabytes of unrelated build hooks.
    excludes=[
        'email_secrets', 'IPython', 'matplotlib', 'pytest', 'pygame',
        'numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'librosa',
        'numba', 'llvmlite', 'torch', 'transformers', 'tensorflow',
        'cupy', 'dask', 'PIL', 'soundfile', 'sounddevice',
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI Music Detector Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
