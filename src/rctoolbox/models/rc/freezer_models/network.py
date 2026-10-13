"""Configurable freezer RC network (the "superstructure").

Every model in ``freezer_models`` is one configuration of this network; the
model packages fix the configuration and document it (``model.py``,
``derivation.ipynb``).

Nodes (state order): air ``T_i`` (measured, always present), and optionally
internal mass ``T_m``, evaporator coil ``T_e`` and wall ``T_w``.

    C_i dT_i/dt = (T_a - T_i)/R_ia                    direct path to the cold room
                + (T_w - T_i)/R_iw                    [wall node]
                + (T_m - T_i)/R_im                    [mass node]
                + g(s) (T_e - T_i)                    [coil node]
                + g(s) (T_ev_meas - T_i)              [measured coil temperature as input]
                + (T_out - T_i)/R_iout                [outdoor link]
                + (T_soil - T_i)/R_isoil              [ground link]
                + Q_0                                 [constant gain]
                + (1 - alpha_m - alpha_e) Phi
    C_m dT_m/dt = (T_i - T_m)/R_im + alpha_m Phi
    C_e dT_e/dt = g(s) (T_i - T_e) + (T_a - T_e)/R_ea + alpha_e Phi
    C_w dT_w/dt = (T_a - T_w)/R_wa + (T_i - T_w)/R_iw

    Phi  = Phi_HC (1 + beta (T_out - T_ref)),  Phi_HC = -P_el   (COP model)
    g(s) = s / R_ie_on + (1 - s) / R_ie_off   (fan-gated coupling; s = compressor-on fraction)
           1 / R_ie_on                         (fans always on, ``fan_gated=False``)

With a coil node all cooling goes to the coil (``alpha_e = 1`` by default):
the air is cooled only through the coil, as in a forced-air evaporator. With
the measured coil temperature as input (``ev_input=True``) Phi is not used at
all: the cooling is g(s) (T_ev - T_i), which does not depend on the COP.

``beta`` is linear in the inputs through the extra input column
``Phi_HC_Tout = Phi_HC * (T_out - T_ref)`` (T_ref = 10 °C), so the model stays
linear.
"""
import numpy as np

from rctoolbox.models.rc.freezer_models._base import FreezerRCModel

T_REF = 10.0

ALL_BOUNDS = {
    "R_ia":     (1e-3, 1.0),
    "C_i":      (1e3, 1e7),
    "R_im":     (1e-4, 1.0),
    "C_m":      (1e4, 1e8),
    "R_ie_on":  (1e-4, 1.0),
    "R_ie_off": (1e-3, 10.0),
    "C_e":      (1e2, 1e6),
    "R_ea":     (1e-2, 100.0),
    "R_iw":     (1e-4, 1.0),
    "R_wa":     (1e-3, 1.0),
    "C_w":      (1e4, 1e8),
    "R_iout":   (1e-2, 100.0),
    "R_isoil":  (1e-2, 100.0),
    "Q_0":      (-500.0, 1000.0),
    "beta":     (-0.1, 0.1),
    "alpha_m":  (0.0, 1.0),
}

LATEX = {
    "R_ia": r"$R_{ia}$", "C_i": r"$C_i$", "R_im": r"$R_{im}$", "C_m": r"$C_m$",
    "R_ie_on": r"$R_{ie}^{on}$", "R_ie_off": r"$R_{ie}^{off}$", "C_e": r"$C_e$", "R_ea": r"$R_{ea}$",
    "R_iw": r"$R_{iw}$", "R_wa": r"$R_{wa}$", "C_w": r"$C_w$", "R_iout": r"$R_{i,out}$",
    "R_isoil": r"$R_{i,soil}$", "Q_0": r"$Q_0$", "beta": r"$\beta$", "alpha_m": r"$\alpha_m$",
}


def network_spec(mass=False, coil=False, wall=False, fan_gated=True, coil_to_ta=False,
                 ev_input=False, link_out=False, link_soil=False, gain=False, cop_tout=False,
                 alpha_m=False, direct_ia=True, adapt_gain=False, adapt_cop=False):
    """Structure dict -> (STATE_NAMES, INPUT_NAMES, DEFAULT_BOUNDS, SWITCHED)."""
    states = ["T_i"] + (["T_m"] if mass else []) + (["T_e"] if coil else []) + (["T_w"] if wall else [])
    states += (["Q_d"] if adapt_gain else []) + (["K_d"] if adapt_cop else [])
    inputs = ["T_a"]
    if not ev_input:
        inputs = ["Phi_HC"] + inputs
    if cop_tout:
        inputs.append("Phi_HC_Tout")
    if ev_input:
        inputs.append("T_ev")
    if link_out:
        inputs.append("T_out")
    if link_soil:
        inputs.append("T_soil")
    if gain:
        inputs.append("unit")
    p = ["C_i"] + (["R_ia"] if direct_ia else [])
    if mass:
        p += ["R_im", "C_m"]
    if coil or ev_input:
        p += ["R_ie_on"] + (["R_ie_off"] if fan_gated else [])
    if coil:
        p += ["C_e"] + (["R_ea"] if coil_to_ta else [])
    if wall:
        p += ["R_iw", "R_wa", "C_w"]
    if link_out:
        p.append("R_iout")
    if link_soil:
        p.append("R_isoil")
    if gain:
        p.append("Q_0")
    if cop_tout:
        p.append("beta")
    if alpha_m:
        p.append("alpha_m")
    switched = ((coil or ev_input) and fan_gated) or adapt_cop
    return tuple(states), tuple(inputs), {k: ALL_BOUNDS[k] for k in p}, switched


class FreezerNetwork(FreezerRCModel):
    """Configurable network; subclasses set ``SPEC`` (kwargs of ``network_spec``)."""
    SPEC: dict = {}
    PARAMS_LATEX = LATEX
    OPTIONAL_PARAMS = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.SPEC is not None:
            cls.STATE_NAMES, cls.INPUT_NAMES, cls.DEFAULT_BOUNDS, cls.SWITCHED = network_spec(**cls.SPEC)

    def set_params(self, params):
        super().set_params(params)
        self._P = self.params

    def calc_matrices(self, s=0.0):
        sp = self.SPEC
        st, inp = self.STATE_NAMES, self.INPUT_NAMES
        n, m = len(st), len(inp)
        ix = {k: i for i, k in enumerate(st)}
        iu = {k: j for j, k in enumerate(inp)}
        P = self._P
        G = np.zeros((n, n))          # conductances between nodes
        Gu = np.zeros((n, m))         # conductances node <- temperature input
        Bq = np.zeros((n, m))         # direct heat inputs (W per unit input)
        C = np.zeros(n)

        def link(a, b, R):
            G[ix[a], ix[b]] += 1.0 / R
            G[ix[b], ix[a]] += 1.0 / R

        def drive(a, u, g):
            Gu[ix[a], iu[u]] += g

        C[ix["T_i"]] = P["C_i"]
        if sp.get("direct_ia", True):
            drive("T_i", "T_a", 1.0 / P["R_ia"])
        if sp.get("fan_gated", True):
            g_ie = s / P.get("R_ie_on", np.inf) + (1.0 - s) / P.get("R_ie_off", np.inf)
        else:
            g_ie = 1.0 / P.get("R_ie_on", np.inf)
        if sp.get("mass"):
            C[ix["T_m"]] = P["C_m"]
            link("T_i", "T_m", P["R_im"])
        if sp.get("coil"):
            C[ix["T_e"]] = P["C_e"]
            G[0, ix["T_e"]] += g_ie
            G[ix["T_e"], 0] += g_ie
            if sp.get("coil_to_ta"):
                drive("T_e", "T_a", 1.0 / P["R_ea"])
        if sp.get("ev_input"):
            drive("T_i", "T_ev", g_ie)
        if sp.get("wall"):
            C[ix["T_w"]] = P["C_w"]
            link("T_i", "T_w", P["R_iw"])
            drive("T_w", "T_a", 1.0 / P["R_wa"])
        if sp.get("link_out"):
            drive("T_i", "T_out", 1.0 / P["R_iout"])
        if sp.get("link_soil"):
            drive("T_i", "T_soil", 1.0 / P["R_isoil"])
        if sp.get("gain"):
            Bq[0, iu["unit"]] += P["Q_0"]
        if not sp.get("ev_input"):
            a_m = P.get("alpha_m", 0.0) if sp.get("mass") else 0.0
            share = np.zeros(n)
            if sp.get("coil"):
                share[ix["T_e"]] = 1.0
            else:
                share[0] = 1.0 - a_m
                if sp.get("mass"):
                    share[ix["T_m"]] = a_m
            Bq[:, iu["Phi_HC"]] += share
            if sp.get("cop_tout"):
                Bq[:, iu["Phi_HC_Tout"]] += P["beta"] * share

        for k in ("Q_d", "K_d"):
            if k in ix:
                C[ix[k]] = 1.0                # constant (random-walk) disturbance states
        A = (G - np.diag(G.sum(axis=1) + Gu.sum(axis=1))) / C[:, None]
        B = (Gu + Bq) / C[:, None]
        # adaptive disturbance states, estimated by the Kalman filter from the
        # lag window and held constant over the forecast:
        #   Q_d [W]  extra heat gain on the air node
        #   K_d [W]  extra cooling while the compressor runs (capacity correction),
        #            acting where Phi acts (coil node, else air)
        if "Q_d" in ix:
            A[0, ix["Q_d"]] = 1.0 / C[0]
        if "K_d" in ix:
            j = ix["T_e"] if sp.get("coil") else 0
            A[j, ix["K_d"]] = -s / C[j]
        return A, B


def make_network(name, **spec):
    """Create a FreezerNetwork subclass for the given structure (research use)."""
    return type(name, (FreezerNetwork,), {"SPEC": dict(spec)})
