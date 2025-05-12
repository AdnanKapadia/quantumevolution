from setuptools import setup

APP = ['app_plotly.py']
DATA_FILES = ['thing.py']
OPTIONS = {
    'argv_emulation': True,
    'packages': ['streamlit', 'numpy', 'plotly', 'matplotlib', 'scipy', 'tqdm'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)