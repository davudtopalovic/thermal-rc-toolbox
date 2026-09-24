"""Parameter space for the 6R2C model.

Free parameters (8): A_f, H_em, H_ww, H_w, H_ve, c_f, phi_int_f, k_sol.

Bounds for A_f, H_em, H_w, H_ve, c_f, phi_int_f, k_sol are computed exactly like
in the 5R1C parameter space. The new parameter H_ww (inner-wall conductance) is
bounded relative to H_em: a multi-layer wall's inner-layer conductance is
typically of the same order as the outer opaque envelope conductance, so a
generous [0.5 * H_em_min, 5 * H_em_max] range is used.
"""
from rctoolbox.models.rc.five_r1c_eniso13790.parameter_space import RCParameterSpace


class RC6R2CParameterSpace(RCParameterSpace):
    def get_free_params(self):
        # Same as 5R1C plus H_ww.
        self.free_params = ['A_f', 'H_em', 'H_ww', 'H_w', 'H_ve', 'c_f', 'phi_int_f', 'k_sol']

    def get_free_param_bounds(self):
        # Delegate to the 5R1C bound calculation (all shared params) and add H_ww.
        super().get_free_param_bounds()

        h_em_lo, h_em_hi = self.free_param_bounds['H_em']
        # Wider than H_em since inner wall interfaces are less physically constrained
        # by "wall area * U-value" and can span a wider range in practice.
        self.free_param_bounds['H_ww'] = [round(0.5 * h_em_lo, 2), round(5.0 * h_em_hi, 2)]

        # Rebuild ordered mirror of names to include H_ww at the right position.
        self.free_param_names = list(self.free_params)
        # Reorder the bounds dict to match self.free_params ordering exactly.
        self.free_param_bounds = {k: self.free_param_bounds[k] for k in self.free_params}

    def get_params_latex(self):
        super().get_params_latex()
        self.params_latex['H_ww'] = r'$H_{ww}$'
