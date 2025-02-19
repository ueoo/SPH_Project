from functools import reduce

import numpy as np
import taichi as ti
import trimesh as tm

from ..utils import SimConfig
from .base_container import BaseContainer


@ti.data_oriented
class PCISPHContainer(BaseContainer):
    def __init__(self, config: SimConfig, GGUI=False):
        super().__init__(config, GGUI)
        # PCISPH related property
        self.density_error = ti.field(dtype=float, shape=())
        self.pcisph_k = ti.field(dtype=float, shape=())
        self.particle_pressure_accelerations = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_predicted_velocities = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_predicted_positions = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.particle_densities_star = ti.field(dtype=float, shape=self.particle_max_num)
