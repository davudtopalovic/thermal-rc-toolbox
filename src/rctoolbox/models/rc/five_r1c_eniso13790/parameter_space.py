from rctoolbox.identification.parameter_space import BaseParameterSpace


class RCParameterSpace(BaseParameterSpace):
    def __init__(self, building_config):

        # ---------------------------
        # Building config
        # ---------------------------
        self.building_config = building_config

        if "max_HP_power" not in self.building_config:
            raise ValueError("building_config must contain 'max_HP_power'")

        # ---------------------------
        # Initialize
        # ---------------------------
        self.get_HP_specifics()
        self.get_free_params()
        self.get_Af_n_floors()
        self.calculate_physical_param_bounds(self.JPD, self.exact_A_f_bounds)
        self.get_free_param_bounds()
        self.get_params_latex()

    @classmethod
    def from_building_config(cls, building_config: dict) -> "RCParameterSpace":
        return cls(building_config)

    # ---------------------------
    # Free parameters
    # ---------------------------
    def get_free_params(self):
        self.free_params = ['A_f', 'H_em', 'H_w', 'H_ve', 'c_f', 'phi_int_f', 'k_sol']

    # ---------------------------
    # Heat pump specifics
    # ---------------------------
    def get_HP_specifics(self):
        # self.hc = self.building_config.get("heat_circuit")
        self.max_HP_power = self.building_config.get("max_HP_power")

    # ---------------------------
    # Building geometry input
    # ---------------------------
    def get_Af_n_floors(self):
        self.JPD = self.building_config.get("JPD")
        self.exact_A_f_bounds = self.building_config.get("exact_A_f_bounds")

    # ---------------------------
    # Physics bounds
    # ---------------------------
    def calculate_physical_param_bounds(self, javni_pogled_data=None, exact_A_f_bounds=None):

        if javni_pogled_data:
            A_f_min = (0.5 * javni_pogled_data["n_floors"]) * javni_pogled_data["A_f"] # A_f_min = 0.5 * n_floors * A_f, i.e. at least 50% of the total floor area (all floors) is considered as the building conditioned zone. 
            A_f_max = (javni_pogled_data["n_floors"] + 0.3) * javni_pogled_data["A_f"] # A_f_max = (n_floors + 0.3) * A_f, i.e. at most 130% of the total floor area (all floors) is considered as the building conditioned zone. The extra 30% is to allow for some cases where the conditioned zone might be larger than the total floor area due to e.g. large balconies or terraces that are included in the conditioned zone.

        elif exact_A_f_bounds:
            A_f_min, A_f_max = exact_A_f_bounds
        else:
            raise ValueError("Either javni_pogled_data or exact_A_f_bounds must be provided in the building config.")

        # constants
        rho_a_c_a = 1.2e3
        F_w = 0.9
        self.bldg_phys_param_constants = {"rho_a_c_a": rho_a_c_a, "F_w": F_w}

        # base ranges
        h_min, h_max = 2.2, 5
        c_f_min, c_f_max = 0.1e5, 5e5
        u_opaque_min, u_opaque_max = 0.1, 2
        u_windows_min, u_windows_max = 0.7, 5
        ach_tot_min, ach_tot_max = 0.1, 1
        ww_ratio_min, ww_ratio_max = 0.05, 0.5
        Phi_int_per_Af_min, Phi_int_per_Af_max = 3, 9 # EN ISO 13790 Table 9.8
        g_n_min, g_n_max = 0.5, 0.85 # EN ISO 13790 Table 9.2 (lower bound is expanded to allow for some cases with very low g_n, e.g. due to shading)

        Vr_min = self.calc_Vr_Ao_Ag(A_f_min, h_min, ww_ratio_min)
        Vr_max = self.calc_Vr_Ao_Ag(A_f_max, h_max, ww_ratio_max)

        bppb = {
            "A_f": [A_f_min, A_f_max],
            "h": [h_min, h_max],
            "c_f": [c_f_min, c_f_max],
            "u_opaque": [u_opaque_min, u_opaque_max],
            "u_windows": [u_windows_min, u_windows_max],
            "ach_tot": [ach_tot_min, ach_tot_max],
            "room_vol": [Vr_min["room_vol"], Vr_max["room_vol"]],
            "A_opaque": [Vr_min["A_opaque"], Vr_max["A_opaque"]],
            "A_glazed": [Vr_min["A_glazed"], Vr_max["A_glazed"]],
            "ww_ratio": [ww_ratio_min, ww_ratio_max],
            "Phi_int": [Phi_int_per_Af_min * A_f_min, Phi_int_per_Af_max * A_f_max],
            "Phi_int_per_Af": [Phi_int_per_Af_min, Phi_int_per_Af_max],
            "g_n": [g_n_min, g_n_max],
        }

        self.bldg_phys_param_bounds = {
            k: [round(v[0], 2), round(v[1], 2)] for k, v in bppb.items()
        }

    # ---------------------------
    # Derived geometry
    # ---------------------------
    def calc_Vr_Ao_Ag(self, A_f, h, ww_ratio):
        A_walls = 4 * (A_f ** 0.5) * h
        A_glazed = A_walls * ww_ratio
        A_opaque = (A_walls - A_glazed) + 2 * A_f
        room_vol = h * A_f
        return {"room_vol": room_vol, "A_opaque": A_opaque, "A_glazed": A_glazed}

    # ---------------------------
    # Free parameter bounds
    # ---------------------------
    def get_free_param_bounds(self):

        bounds = {}
        rho_a_c_a = self.bldg_phys_param_constants["rho_a_c_a"]
        F_w = self.bldg_phys_param_constants["F_w"]
        bppb = self.bldg_phys_param_bounds

        bounds['A_f'] = bppb["A_f"]

        h_em = [bppb["A_opaque"][0]*bppb["u_opaque"][0],
                bppb["A_opaque"][1]*bppb["u_opaque"][1]]

        h_w = [bppb["A_glazed"][0]*bppb["u_windows"][0],
               bppb["A_glazed"][1]*bppb["u_windows"][1]]

        h_ve = [rho_a_c_a*bppb["room_vol"][0]*bppb["ach_tot"][0]/3600,
                rho_a_c_a*bppb["room_vol"][1]*bppb["ach_tot"][1]/3600]

        
        bounds['H_em'] = h_em
        bounds['H_w']  = h_w
        bounds['H_ve'] = h_ve
        
        bounds['c_f'] = bppb["c_f"]
        bounds['phi_int_f'] = bppb["Phi_int_per_Af"]

        k_sol = [
            F_w * bppb["g_n"][0] * bppb["A_glazed"][0],
            F_w * bppb["g_n"][1] * bppb["A_glazed"][1]
        ]
        bounds['k_sol'] = k_sol

        self.free_param_bounds = {
            k: [round(v[0], 2), round(v[1], 2)] for k, v in bounds.items()
        }
        # Mirror keys onto free_param_names for the BaseParameterSpace contract.
        self.free_param_names = list(self.free_param_bounds)

    # ---------------------------
    # Latex labels
    # ---------------------------
    def get_params_latex(self):
        self.params_latex = {
            'A_f': r'$A_f$',
            'H_em': r'$H_{em}$',
            'H_w': r'$H_w$',
            'H_ve': r'$H_{ve}$',
            'c_f': r'$c_f$',
            'phi_int_f': r'$\phi_{int,f}$',
            'k_sol': r"$k_{sol}$"
        }


# Forward-compat alias for clarity once 2R2C lands. Both names point to the
# same class; existing code that imports RCParameterSpace keeps working.
RC5R1CParameterSpace = RCParameterSpace