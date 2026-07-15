import numpy as np


def tri_to_quadrature(tri, output_amp: float = 0.8):
    """
    Given a triangle wave with fixed amplitude, but varying frequency,
    return a sine, cosine wave pair with the same phase and frequency at output_amp
    """
    tri = np.asarray(tri, dtype=np.float32)

    # Normalize to roughly [-1, 1] even if capture amplitude/offset is not exact.
    eps = 1e-7
    x_min = float(np.min(tri))
    x_max = float(np.max(tri))
    amp = max(0.5 * (x_max - x_min), eps)
    mid = 0.5 * (x_max + x_min)
    u = np.clip((tri - mid) / amp, -1.0, 1.0)

    # sin(phi) from normalized triangle mapping.
    theta = 0.5 * np.pi * u
    out_sin = np.sin(theta)
    cos_mag = np.sqrt(np.maximum(0.0, 1.0 - out_sin * out_sin))

    # sign(dtri/dt) ~= sign(cos(phi)); only allow sign flips near |u| ~ 1.
    slope = np.gradient(u)
    slope_sign = np.sign(slope)
    slope_sign[slope_sign == 0.0] = 1.0

    sign = np.empty_like(slope_sign)
    sign[0] = slope_sign[0]
    flip_gate = 0.9
    for i in range(1, slope_sign.shape[0]):
        prev = sign[i - 1]
        cand = slope_sign[i]
        if cand != prev and abs(u[i]) >= flip_gate:
            sign[i] = cand
        else:
            sign[i] = prev

    out_cos = sign * cos_mag

    out_sin *= output_amp
    out_cos *= output_amp

    return out_sin, out_cos
