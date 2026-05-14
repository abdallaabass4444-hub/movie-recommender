"""
run.py — Launch the Movie Recommendation Streamlit app.

Usage:
    python run.py

Make sure you have installed the requirements first:
    pip install -r requirements.txt

Your data folder must contain:
    data/movies.csv
    data/ratings.csv
"""

import subprocess
import sys
import os

APP_FILE = os.path.join(os.path.dirname(__file__), 'app.py')


def main():
    print('🎬 Starting Movie Recommendation System...')
    print('   Opening at http://localhost:8501')
    print('   Press Ctrl+C to stop.\n')

    cmd = [sys.executable, '-m', 'streamlit', 'run', APP_FILE,
           '--server.port', '8501',
           '--server.headless', 'false']
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print('\n👋 App stopped.')
    except FileNotFoundError:
        print('❌ Streamlit not found. Run: pip install streamlit')
        sys.exit(1)


if __name__ == '__main__':
    main()
