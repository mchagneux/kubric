# Copyright 2020 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import traitlets as tl

from kubric.core import color
from kubric.core import base


__all__ = ("Vector3D", "Scale", "Quaternion", "RGB", "RGBA", "AssetInstance")


class Vector3D(tl.TraitType):
  default_value = np.zeros(shape=[3], dtype=np.float32)
  info_text = "a 3D vector of floats"

  def validate(self, obj, value):
    value = np.array(value, dtype=np.float32)
    if value.shape != (3,):
      self.error(obj, value)
    else:
      return value


class Scale(tl.TraitType):
  default_value = np.ones(shape=[3], dtype=np.float32)
  info_text = "a 3D vector of floats"

  def validate(self, obj, value):
    value = np.array(value, dtype=np.float32)
    if value.shape == ():  # broadcast scalar Scale to 3D
      value = np.array([value, value, value], dtype=np.float32)
    elif value.shape == (1,):  # broadcast scalar Scale to 3D
      value = np.array([value[0], value[0], value[0]], dtype=np.float32)
    if value.shape != (3,):
      self.error(obj, value)
    else:
      return value


class Quaternion(tl.TraitType):
  default_value = np.array([1, 0, 0, 0], dtype=np.float32)
  info_text = "a 4D vector (WXYZ quaternion) of floats"

  def validate(self, obj, value):
    value = np.array(value, dtype=np.float32)
    if value.shape != (4,):
      self.error(obj, value)
    else:
      return value


class RGBA(tl.TraitType):
  default_value = color.Color(0., 0., 0., 1.0)
  info_text = "an RGBA color"

  def validate(self, obj, value):
    if isinstance(value, color.Color):
      rgba = value
    elif isinstance(value, int):
      rgba = color.Color.from_hexint(value)
    elif isinstance(value, str):
      rgba = color.Color.from_hexstr(value)
    elif len(value) in [3, 4]:
      rgba = color.Color(*value)
    else:
      return self.error(obj, value)

    if not all([0 <= x <= 1 for x in rgba]):
      self.error(obj, value)

    return rgba


# TODO: it is inconsistent to use Color object for RGBA and a regular tuple for RGB.
#       But we do need both types. So maybe we should have both ColorRGBA and ColorRGB classes?
class RGB(tl.TraitType):
  default_value = (0., 0., 0.)
  info_text = "an RGB color"

  def validate(self, obj, value):
    if isinstance(value, color.Color):
      rgb = value.rgb
    elif isinstance(value, int):
      rgb = color.Color.from_hexint(value).rgb
    elif isinstance(value, str):
      rgb = color.Color.from_hexstr(value).rgb
    elif len(value) == 3:
      rgb = color.Color(*value).rgb
    else:
      return self.error(obj, value)

    if not all([0 <= x <= 1 for x in rgb]):
      self.error(obj, value)

    return rgb


class AssetInstance(tl.Instance):
  default_value = base.UndefinedAsset()

  def validate(self, obj: base.Asset, value):
    super().validate(obj, value)

    # make sure the new asset is part of all scenes that the parent is part of
    for scene in obj.scenes:
      scene.add(value)

    return value
