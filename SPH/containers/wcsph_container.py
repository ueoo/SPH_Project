from functools import reduce

import numpy as np
import taichi as ti
import trimesh as tm

from ..utils import SimConfig
from .base_container import BaseContainer


@ti.data_oriented
class WCSPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
