#!/usr/bin/env python3
"""
Decompose a 2D image into circular harmonics (angular Fourier modes) multiplied
by radial basis functions (Bessel functions) as popularized in Harmonic
Networks (Worrall et al., 2017).

The script performs the following steps:

1. Load a grayscale image and normalize it to [0, 1].
2. Sample the image in polar coordinates (radius, angle) about its center.
3. Compute circular harmonic (angular) coefficients via a Fourier transform.
4. Project the radial profiles onto a truncated set of Bessel radial basis
   functions (Fourier-Bessel expansion).
5. Reconstruct the contribution of each (m, n) component on the Cartesian grid,
   save the strongest components as RGBA images (alpha set by |coefficient|),
   and report reconstruction error statistics.

Only the decomposition is implemented; animating the components is intentionally
left out so that the generated assets can be consumed by a separate animation
pipeline.

Example usage:
    python circular_harmonics.py \
        --image ../../assets/blobs.png \
        --resize 256 256 \
        --max-order 6 \
        --radial-modes 10 \
        --num-angles 512 \
        --num-radii 256 \
        --output-dir ../../artifacts/circular_harmonics \
        --save-top 24
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jn_zeros, jv

ArrayLike = np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose an image into circular harmonics + radial Bessel modes."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image (any format readable by Pillow).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optionally resize the image before processing.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=512,
        help="Number of angular samples used for the polar grid.",
    )
    parser.add_argument(
        "--num-radii",
        type=int,
        default=256,
        help="Number of radial samples used for the polar grid.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=6,
        help="Maximum circular harmonic order |m| to retain.",
    )
    parser.add_argument(
        "--radial-modes",
        type=int,
        default=12,
        help="Number of radial Bessel modes per order to project onto.",
    )
    parser.add_argument(
        "--coeff-threshold",
        type=float,
        default=1e-4,
        help="Skip components whose coefficient magnitude is below this value.",
    )
    parser.add_argument(
        "--save-top",
        type=int,
        default=24,
        help="Number of strongest components to export as RGBA images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./circular_harmonics_output"),
        help="Directory where reconstructions, components, and metadata are stored.",
    )
    parser.add_argument(
        "--no-center-crop",
        action="store_true",
        help=(
            "By default only the largest inscribed circle is modeled. "
            "Set this flag to keep the full image (values outside the circle remain zero)."
        ),
    )
    return parser.parse_args()


def load_grayscale_image(image_path: Path, resize: Tuple[int, int] | None) -> ArrayLike:
    img = Image.open(image_path).convert("L")
    if resize:
        img = img.resize(tuple(resize), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float64)
    return arr / 255.0


def build_regular_interpolator(image: ArrayLike) -> RegularGridInterpolator:
    h, w = image.shape
    grid_y = np.arange(h, dtype=np.float64)
    grid_x = np.arange(w, dtype=np.float64)
    return RegularGridInterpolator(
        (grid_y, grid_x),
        image,
        bounds_error=False,
        fill_value=0.0,
    )


def cartesian_to_polar(
    image: ArrayLike, num_radii: int, num_angles: int
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float, Tuple[float, float]]:
    h, w = image.shape
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    max_radius = min(center_y, center_x)
    radii = np.linspace(0.0, max_radius, num_radii, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num_angles, endpoint=False, dtype=np.float64)

    rr, aa = np.meshgrid(radii, angles, indexing="xy")
    sample_y = center_y + rr * np.sin(aa)
    sample_x = center_x + rr * np.cos(aa)
    interpolator = build_regular_interpolator(image)
    polar = interpolator(
        np.stack([sample_y.ravel(), sample_x.ravel()], axis=-1)
    ).reshape(num_angles, num_radii)
    return polar, radii, angles, max_radius, (center_y, center_x)


def angular_profiles(polar_values: ArrayLike, max_order: int) -> Dict[int, ArrayLike]:
    num_angles = polar_values.shape[0]
    if max_order >= num_angles // 2:
        raise ValueError(
            "max_order must be < num_angles/2 to avoid aliasing. "
            f"Received max_order={max_order}, num_angles={num_angles}."
        )
    fourier = np.fft.fft(polar_values, axis=0) / num_angles
    spacing = 1.0 / num_angles
    orders = np.fft.fftfreq(num_angles, d=spacing)
    profiles: Dict[int, ArrayLike] = {}
    for idx, m in enumerate(orders):
        m_int = int(np.rint(m))
        if -max_order <= m_int <= max_order:
            profiles[m_int] = fourier[idx]
    return profiles


@dataclass
class BesselRadialBasis:
    order: int
    zeros: ArrayLike
    norms: ArrayLike
    samples: ArrayLike  # (num_modes, num_radii)

    @classmethod
    def build(cls, order: int, num_modes: int, radii_norm: ArrayLike) -> "BesselRadialBasis":
        zeros = jn_zeros(order, num_modes)
        samples = []
        norms = []
        for zero in zeros:
            raw = jv(order, zero * radii_norm)
            weight = np.trapezoid((raw**2) * radii_norm, radii_norm)
            norm = math.sqrt(weight) + 1e-12
            samples.append(raw / norm)
            norms.append(norm)
        return cls(order=order, zeros=zeros, norms=np.asarray(norms), samples=np.asarray(samples))

    def evaluate_on_grid(self, r_norm_grid: ArrayLike, mode_idx: int) -> ArrayLike:
        zero = self.zeros[mode_idx]
        norm = self.norms[mode_idx]
        raw = jv(self.order, zero * r_norm_grid)
        raw = np.where(r_norm_grid <= 1.0, raw, 0.0)
        return raw / norm


def project_radial_profile(
    profile: ArrayLike, basis: BesselRadialBasis, radii_norm: ArrayLike
) -> ArrayLike:
    coeffs = np.empty(basis.samples.shape[0], dtype=np.complex128)
    for idx, samples in enumerate(basis.samples):
        integrand = profile * np.conj(samples) * radii_norm
        coeffs[idx] = np.trapezoid(integrand, radii_norm)
    return coeffs


def cartesian_grids(
    image_shape: Tuple[int, int],
    center: Tuple[float, float],
    max_radius: float,
    keep_outside: bool,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    h, w = image_shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    dy = yy - center[0]
    dx = xx - center[1]
    radii = np.sqrt(dx**2 + dy**2)
    r_norm = radii / max_radius
    theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    if keep_outside:
        mask = np.ones_like(r_norm, dtype=bool)
    else:
        mask = r_norm <= 1.0
    return r_norm, theta, mask


@dataclass
class HarmonicComponent:
    index: int
    m: int
    order: int
    mode: int
    coefficient: complex


def evaluate_component(
    component: HarmonicComponent,
    basis_cache: Dict[int, BesselRadialBasis],
    r_norm_grid: ArrayLike,
    theta_grid: ArrayLike,
    mask: ArrayLike,
    radial_field_cache: Dict[Tuple[int, int], ArrayLike],
    angular_field_cache: Dict[int, ArrayLike],
) -> ArrayLike:
    radial_key = (component.order, component.mode)
    if radial_key not in radial_field_cache:
        radial_vals = basis_cache[component.order].evaluate_on_grid(
            r_norm_grid, component.mode
        )
        radial_field_cache[radial_key] = radial_vals
    radial_vals = radial_field_cache[radial_key]

    if component.m not in angular_field_cache:
        angular_field_cache[component.m] = np.exp(1j * component.m * theta_grid)
    angular_vals = angular_field_cache[component.m]

    field = component.coefficient * radial_vals * angular_vals
    real_field = np.real(field)
    return np.where(mask, real_field, 0.0)


def normalize_for_rgba(field: ArrayLike, alpha: float, mask: ArrayLike) -> ArrayLike:
    cropped = np.where(mask, field, 0.0)
    span = cropped.max() - cropped.min()
    if span < 1e-9:
        scaled = np.zeros_like(cropped) + 0.5
    else:
        scaled = (cropped - cropped.min()) / span
    rgba = np.zeros((*field.shape, 4), dtype=np.float32)
    for ch in range(3):
        rgba[..., ch] = scaled
    rgba[..., 3] = alpha * mask.astype(np.float32)
    return np.clip(rgba * 255.0, 0, 255).astype(np.uint8)


def save_component_image(path: Path, rgba_array: ArrayLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_array, mode="RGBA").save(path)


def summarize_components(components: Iterable[HarmonicComponent]) -> List[Dict[str, float]]:
    summary = []
    for comp in components:
        coeff = comp.coefficient
        summary.append(
            {
                "index": comp.index,
                "m": comp.m,
                "radial_mode": comp.mode + 1,
                "abs_coeff": float(np.abs(coeff)),
                "phase_deg": float(np.degrees(np.angle(coeff))),
                "real": float(np.real(coeff)),
                "imag": float(np.imag(coeff)),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    image = load_grayscale_image(args.image, args.resize)
    polar, radii, _, max_radius, center = cartesian_to_polar(
        image, args.num_radii, args.num_angles
    )
    radii_norm = radii / max_radius

    profiles = angular_profiles(polar, args.max_order)
    radial_basis_cache: Dict[int, BesselRadialBasis] = {}
    components: List[HarmonicComponent] = []
    component_index = 0

    for m in sorted(profiles.keys()):
        order = abs(m)
        if order not in radial_basis_cache:
            radial_basis_cache[order] = BesselRadialBasis.build(
                order, args.radial_modes, radii_norm
            )
        coeffs = project_radial_profile(profiles[m], radial_basis_cache[order], radii_norm)
        for mode_idx, coeff in enumerate(coeffs):
            if np.abs(coeff) < args.coeff_threshold:
                continue
            components.append(
                HarmonicComponent(
                    index=component_index,
                    m=m,
                    order=order,
                    mode=mode_idx,
                    coefficient=coeff,
                )
            )
            component_index += 1

    if not components:
        raise RuntimeError(
            "No harmonic components exceeded the coefficient threshold. "
            "Consider lowering --coeff-threshold or increasing --max-order/--radial-modes."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_dir = args.output_dir / "components"

    max_coeff = max(float(np.abs(comp.coefficient)) for comp in components)
    components_sorted = sorted(
        components, key=lambda c: float(np.abs(c.coefficient)), reverse=True
    )
    top_k = args.save_top if args.save_top is not None else len(components_sorted)
    selected_ids = {comp.index for comp in components_sorted[:top_k]}

    r_norm_grid, theta_grid, mask = cartesian_grids(
        image.shape, center, max_radius, args.no_center_crop
    )
    target_image = image if args.no_center_crop else np.where(mask, image, 0.0)

    reconstruction = np.zeros_like(image)
    radial_field_cache: Dict[Tuple[int, int], ArrayLike] = {}
    angular_field_cache: Dict[int, ArrayLike] = {}

    for comp in components:
        contribution = evaluate_component(
            comp,
            radial_basis_cache,
            r_norm_grid,
            theta_grid,
            mask,
            radial_field_cache,
            angular_field_cache,
        )
        reconstruction += contribution

        if comp.index in selected_ids:
            alpha = 0.0 if max_coeff < 1e-12 else min(
                float(np.abs(comp.coefficient)) / max_coeff, 1.0
            )
            rgba = normalize_for_rgba(contribution, alpha, mask)
            filename = f"component_m{comp.m:+d}_n{comp.mode + 1:02d}.png"
            save_component_image(component_dir / filename, rgba)

    residual = target_image - reconstruction
    rmse = float(np.sqrt(np.mean(residual**2)))
    rel_rmse = rmse / (np.sqrt(np.mean(target_image**2)) + 1e-12)
    linf = float(np.max(np.abs(residual)))

    np_clip = lambda arr: np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(np_clip(target_image), mode="L").save(args.output_dir / "target.png")
    Image.fromarray(np_clip(reconstruction), mode="L").save(
        args.output_dir / "reconstruction.png"
    )
    residual_range = np.ptp(residual)
    residual_vis = (residual - residual.min()) / (residual_range + 1e-12)
    Image.fromarray(np_clip(residual_vis), mode="L").save(args.output_dir / "residual.png")

    summary = {
        "image": str(args.image),
        "resize": args.resize,
        "num_angles": args.num_angles,
        "num_radii": args.num_radii,
        "max_order": args.max_order,
        "radial_modes": args.radial_modes,
        "coeff_threshold": args.coeff_threshold,
        "num_components": len(components),
        "rmse": rmse,
        "relative_rmse": float(rel_rmse),
        "linf_error": linf,
        "max_coeff": max_coeff,
        "components": summarize_components(components),
        "saved_components": [comp.index for comp in components_sorted[:top_k]],
    }

    with open(args.output_dir / "decomposition_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Decomposition complete: {len(components)} components "
        f"(saved top {len(selected_ids)}) | RMSE={rmse:.4e} | rel RMSE={rel_rmse:.4e}"
    )


if __name__ == "__main__":
    main()

