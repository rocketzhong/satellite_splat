"""
Copied from ns-train, which is used to debug in Pycharm
python it with args should be the same as ns-train
"""

import re
import sys
from nerfstudio.scripts.train import entrypoint
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(entrypoint())