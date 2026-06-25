import pathlib
import sys

# Make cc_palette / transcoder importable from tests regardless of CWD.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
