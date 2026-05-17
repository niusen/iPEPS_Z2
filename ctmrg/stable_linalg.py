from contextlib import contextmanager

import yastn


def _setting_value(setting, name, default):
    if setting is None:
        return default
    return getattr(setting, name, default)


@contextmanager
def _patched_svd_backward(a, ad_decomp_reg):
    if ad_decomp_reg is None:
        yield
        return

    backend = a.config.backend
    original_svd = backend.svd

    def svd_with_reg(data, meta, sizes, fullrank_uv=False, ad_decomp_reg=ad_decomp_reg, diagnostics=None, **kwargs):
        return original_svd(
            data,
            meta,
            sizes,
            fullrank_uv=fullrank_uv,
            ad_decomp_reg=ad_decomp_reg,
            diagnostics=diagnostics,
            **kwargs,
        )

    backend.svd = svd_with_reg
    try:
        yield
    finally:
        backend.svd = original_svd


def stable_svd_with_truncation(a, *args, ctm_setting=None, **kwargs):
    """Run yastn SVD with CTMRG-friendly defaults for AD stability."""
    ad_decomp_reg = _setting_value(ctm_setting, "svd_ad_decomp_reg", 1.0e-8)
    kwargs.setdefault("fix_signs", _setting_value(ctm_setting, "svd_fix_signs", True))
    with _patched_svd_backward(a, ad_decomp_reg):
        return yastn.linalg.svd_with_truncation(a, *args, **kwargs)


def stable_svd(a, *args, ctm_setting=None, **kwargs):
    ad_decomp_reg = _setting_value(ctm_setting, "svd_ad_decomp_reg", 1.0e-8)
    kwargs.setdefault("fix_signs", _setting_value(ctm_setting, "svd_fix_signs", True))
    with _patched_svd_backward(a, ad_decomp_reg):
        return yastn.linalg.svd(a, *args, **kwargs)


def stable_norm_value(a, cutoff=1.0e-30):
    nrm = yastn.linalg.norm(a)
    if hasattr(nrm, "clamp_min"):
        return nrm.clamp_min(cutoff)
    return max(nrm, cutoff)


def stable_normalize(a, cutoff=1.0e-30):
    return a / stable_norm_value(a, cutoff=cutoff)


def projector_rsqrt_cutoff(ctm_setting):
    return max(
        _setting_value(ctm_setting, "CTM_trun_tol", 1.0e-10),
        _setting_value(ctm_setting, "projector_min_singular_cutoff", 1.0e-8),
    )


def stable_projector_rsqrt(s, ctm_setting):
    return s.rsqrt(cutoff=projector_rsqrt_cutoff(ctm_setting))
