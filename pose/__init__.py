from __future__ import absolute_import

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar

from . import datasets
from . import models
from . import utils



__version__ = '0.1.0'