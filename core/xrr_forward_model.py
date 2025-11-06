# MIT License
#
# Copyright (c) 2024 XRR Demo Code
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core computations for the multilayer XRR forward model."""
from __future__ import annotations

import numpy as np

__all__ = [
    "CU_KA_WAVELENGTH",
    "CLASSICAL_E_RADIUS",
    "AVOGADRO",
    "MATERIAL_DB",
    "ABSORPTION_DB",
    "refractive_index_from_density",
    "parratt_reflectivity",
    "assemble_parratt_inputs",
    "simulate_reflectivity_curve",
]

CU_KA_WAVELENGTH = 1.5406  # Å
CLASSICAL_E_RADIUS = 2.8179403262e-5  # Å
AVOGADRO = 6.02214076e23

MATERIAL_DB = {
    "ambient": {"molar_mass": None, "electrons": None, "density_ref": 0.0},
    "film": {"molar_mass": 60.0843, "electrons": 30, "density_ref": 2.20},  # SiO2 baseline
    "substrate": {"molar_mass": 28.0855, "electrons": 14, "density_ref": 2.33},  # Si baseline
}
ABSORPTION_DB = {"ambient": 0.0, "film": 2.5e-7, "substrate": 1.0e-6}


def refractive_index_from_density(density_g_cm3, role, wavelength=CU_KA_WAVELENGTH):
    """Return the complex refractive index 1 - delta + i * beta for a material at a given density."""
    if role == "ambient":
        return 1.0 + 0.0j
    data = MATERIAL_DB[role]
    electron_density_cm3 = density_g_cm3 * AVOGADRO * data["electrons"] / data["molar_mass"]
    electron_density = electron_density_cm3 / 1e24
    delta = (electron_density * CLASSICAL_E_RADIUS * wavelength**2) / (2 * np.pi)
    beta = ABSORPTION_DB[role] * (density_g_cm3 / data["density_ref"])
    return 1 - delta + 1j * beta


def parratt_reflectivity(
    two_theta_deg,
    layer_definitions,
    layer_thicknesses_nm,
    interface_roughness_nm,
    wavelength=CU_KA_WAVELENGTH,
):
    """Parratt recursion with Nevot-Croce roughness correction for specular XRR."""
    theta = np.radians(two_theta_deg / 2)
    k0 = 2 * np.pi / wavelength
    n_list = np.array([refractive_index_from_density(*layer) for layer in layer_definitions], dtype=np.complex128)
    kz = np.vstack([k0 * np.lib.scimath.sqrt(n**2 - np.cos(theta) ** 2) for n in n_list])
    thickness_lookup = np.array([0.0] + [t * 10.0 for t in layer_thicknesses_nm] + [0.0])  # nm → Å
    roughness_lookup = np.array([r * 10.0 for r in interface_roughness_nm])  # nm → Å

    r = np.zeros_like(theta, dtype=np.complex128)
    for i in range(len(n_list) - 1, 0, -1):
        ki = kz[i - 1]
        kj = kz[i]
        r_ij = (ki - kj) / (ki + kj)
        sigma = roughness_lookup[i - 1] if i - 1 < len(roughness_lookup) else 0.0
        if sigma > 0:
            r_ij *= np.exp(-2 * ki * kj * sigma**2)
        phase = np.exp(2j * kj * thickness_lookup[i])
        r = (r_ij + r * phase) / (1 + r_ij * r * phase)
    reflectivity = np.abs(r) ** 2
    return np.clip(reflectivity, 1e-12, None)


def assemble_parratt_inputs(layers, substrate_density, substrate_roughness):
    """Build the tuples required for Parratt recursion based on layer metadata."""
    layer_definitions = [(0.0, "ambient")]
    thicknesses = []
    roughness_terms = []
    for layer in layers:
        if layer["thickness_nm"] <= 0:
            continue
        layer_definitions.append((layer["density"], "film"))
        thicknesses.append(layer["thickness_nm"])
        roughness_terms.append(layer["roughness_nm"])
    layer_definitions.append((substrate_density, "substrate"))
    roughness_terms.append(substrate_roughness)
    return layer_definitions, thicknesses, roughness_terms


def simulate_reflectivity_curve(
    *,
    layers,
    substrate_density,
    substrate_roughness_nm,
    two_theta_min=0.2,
    two_theta_max=6.0,
    points=1500,
    wavelength=CU_KA_WAVELENGTH,
):
    """Compute the reflectivity curve for a multilayer stack over a two-theta sweep."""
    two_theta = np.linspace(two_theta_min, two_theta_max, points)
    layer_definitions, thicknesses, roughness = assemble_parratt_inputs(
        layers, substrate_density, substrate_roughness_nm
    )
    reflectivity = parratt_reflectivity(two_theta, layer_definitions, thicknesses, roughness, wavelength=wavelength)
    return two_theta, reflectivity
