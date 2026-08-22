from math import sqrt
import logging

import torch
from torch.optim.lbfgs import LBFGS, _strong_wolfe


log = logging.getLogger(__name__)


def _scalar_search_armijo(phi, phi0, derphi0, args=(), c1=1e-4, alpha0=1.0, amin=1.0e-8):
    """Derivative-free Armijo backtracking with quadratic/cubic interpolation."""
    log.info("LS expected phi: %s (derphi0: %s)", phi0 + c1 * alpha0 * derphi0, derphi0)
    phi_a0 = phi(alpha0, *args)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0

    alpha1 = -(derphi0) * alpha0**2 / (2.0 * (phi_a0 - phi0 - derphi0 * alpha0))
    phi_a1 = phi(alpha1, *args)
    if phi_a1 <= phi0 + c1 * alpha1 * derphi0:
        return alpha1, phi_a1

    while alpha1 > amin:
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        b = b / factor

        alpha2 = (-b + sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2, *args)
        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    return None, phi_a1


class LBFGS_MOD(LBFGS):
    """PyTorch L-BFGS with derivative-free backtracking and complex-gradient support."""

    def __init__(
        self,
        params,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        history_size=100,
        line_search_fn=None,
        line_search_eps=1.0e-4,
    ):
        super().__init__(
            params,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        assert len(self.param_groups) == 1
        self.param_groups[0]["line_search_eps"] = line_search_eps

    def _directional_evaluate_derivative_free(self, closure, t, x, d):
        self._add_grad(t, d)
        with torch.no_grad():
            orig_loss = closure(True)
        loss = float(orig_loss)
        self._set_param(x)
        return loss

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        with torch.enable_grad():
            loss = float(closure(linesearching=True))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step_2c(self, closure, closure_linesearch):
        """Perform one L-BFGS step.

        ``closure`` computes loss and gradient. ``closure_linesearch`` computes
        only loss under ``torch.no_grad()``, which is useful when the line search
        should reuse expensive CTMRG code without building extra AD graphs.
        """
        assert len(self.param_groups) == 1
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        with torch.enable_grad():
            orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        is_complex = flat_grad.is_complex()
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        if opt_cond:
            return orig_loss

        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = torch.real(y.conj().dot(s)) if is_complex else y.dot(s)
                if ys > 1.0e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = ys / y.conj().dot(y) if is_complex else ys / y.dot(y)

                num_old = len(old_dirs)
                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                q = flat_grad.neg()
                if is_complex:
                    for i in range(num_old - 1, -1, -1):
                        al[i] = torch.real(old_stps[i].conj().dot(q)) * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])
                else:
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_stps[i].dot(q) * ro[i]
                        q.add_(old_dirs[i], alpha=-al[i])

                d = r = torch.mul(q, H_diag)
                if is_complex:
                    for i in range(num_old):
                        be_i = torch.real(old_dirs[i].conj().dot(r)) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)
                else:
                    for i in range(num_old):
                        be_i = old_dirs[i].dot(r) * ro[i]
                        r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            gtd = torch.real(flat_grad.conj().dot(d)) if is_complex else flat_grad.dot(d)
            if gtd > -tolerance_change:
                break

            ls_func_evals = 0
            if line_search_fn is not None and line_search_fn != "default":
                if line_search_fn == "backtracking":
                    x_init = self._clone_param()

                    def obj_func(t_local, x_local, d_local):
                        return self._directional_evaluate_derivative_free(
                            closure_linesearch, t_local, x_local, d_local
                        )

                    t, loss = _scalar_search_armijo(obj_func, loss, gtd, args=(x_init, d), alpha0=t)
                    if t is None:
                        t = group["line_search_eps"]
                        loss = obj_func(t, x_init, d)
                elif line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()

                    def obj_func(x_local, t_local, d_local):
                        return self._directional_evaluate(closure, x_local, t_local, d_local)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd
                    )
                else:
                    raise RuntimeError("unsupported line search")

                log.info("LS final step: %s", t)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                self._add_grad(t, d)
                if n_iter != max_iter:
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            if n_iter == max_iter:
                break
            if current_evals >= max_eval:
                break
            if opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
