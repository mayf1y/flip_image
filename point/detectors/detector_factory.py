from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .point import PointDetector

detector_factory = {
  'point': PointDetector,
}
