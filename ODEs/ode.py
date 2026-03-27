import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


# ============================================================
# TIME CALIBRATION
# All time is in DAYS
#
# 15 minutes  = 15 / (24*60) days = 1/96 days
# 12 hours    = 0.5 days
# 3 weeks     = 21 days
# first chemotherapy = day 21
# ============================================================

DT_15_MIN = 15 / (24 * 60)   # exactly 15 minutes in days

# a = 0.001667 per 15 minutes -> convert to per day
A_PER_DAY = 0.001667 * 96    # = 0.160032 per day

# Common EC50 for all cell types:
# 25.7 ug/mL doxorubicin
EC50_COMMON_uM = 25.7


# ============================================================
# Drug concentration:
# - no treatment before start_day
# - first treatment starts at day 21
# - repeated every 21 days (3 weeks)
# - spike lasts 15 minutes
# - then exponential decay with half-life 12 hours
# ============================================================
def drug_concentration(
    t,
    C_max,
    start_day=21.0,
    every=21.0,
    spike_duration=DT_15_MIN,
    half_life=0.5
):
    if t < start_day:
        return 0.0

    t_since_start = t - start_day
    t_in_cycle = t_since_start % every

    if t_in_cycle < spike_duration:
        return C_max
    else:
        return C_max * 2 ** (-(t_in_cycle - spike_duration) / half_life)


# ============================================================
# Fusion + chemotherapy model
# ============================================================
def simulate_fusion_model_with_decay_chemo(
    R0, G0, DR0, DG0, DP0,
    a, f, K,
    eta, EC50, C_max,
    dt, T,
    chemo_start_day=21.0,
    chemo_every=21.0,
    spike_duration=DT_15_MIN,
    half_life=0.5
):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps + 1)

    R = np.zeros(n_steps + 1)
    G = np.zeros(n_steps + 1)
    DR = np.zeros(n_steps + 1)
    DG = np.zeros(n_steps + 1)
    DP = np.zeros(n_steps + 1)
    C = np.zeros(n_steps + 1)

    R[0], G[0], DR[0], DG[0], DP[0] = R0, G0, DR0, DG0, DP0

    for n in range(n_steps):
        t_n = t[n]

        N = R[n] + G[n] + DR[n] + DG[n] + DP[n]
        RG = R[n] + G[n]

        if N <= 0 or RG <= 0:
            for k in range(n, n_steps + 1):
                C[k] = drug_concentration(
                    t[k],
                    C_max,
                    start_day=chemo_start_day,
                    every=chemo_every,
                    spike_duration=spike_duration,
                    half_life=half_life
                )
            break

        log_term = np.log(N / K)

        C_current = drug_concentration(
            t_n,
            C_max,
            start_day=chemo_start_day,
            every=chemo_every,
            spike_duration=spike_duration,
            half_life=half_life
        )
        C[n] = C_current

        kill_R = eta['R'] * C_current / (C_current + EC50['R'])
        kill_G = eta['G'] * C_current / (C_current + EC50['G'])
        kill_DR = eta['DR'] * C_current / (C_current + EC50['DR'])
        kill_DG = eta['DG'] * C_current / (C_current + EC50['DG'])
        kill_DP = eta['DP'] * C_current / (C_current + EC50['DP'])

        R[n + 1] = R[n] + dt * (
            -a * R[n] * log_term
            - 2 * f * R[n]
            - kill_R * R[n]
        )

        G[n + 1] = G[n] + dt * (
            -a * G[n] * log_term
            - 2 * f * G[n]
            - kill_G * G[n]
        )

        DR[n + 1] = DR[n] + dt * (
            (f * R[n] ** 2) / (2 * RG)
            - a * DR[n] * log_term
            - kill_DR * DR[n]
        )

        DG[n + 1] = DG[n] + dt * (
            (f * G[n] ** 2) / (2 * RG)
            - a * DG[n] * log_term
            - kill_DG * DG[n]
        )

        DP[n + 1] = DP[n] + dt * (
            (2 * f * R[n] * G[n]) / RG
            - a * DP[n] * log_term
            - kill_DP * DP[n]
        )

        R[n + 1] = max(R[n + 1], 0.0)
        G[n + 1] = max(G[n + 1], 0.0)
        DR[n + 1] = max(DR[n + 1], 0.0)
        DG[n + 1] = max(DG[n + 1], 0.0)
        DP[n + 1] = max(DP[n + 1], 0.0)

    C[-1] = drug_concentration(
        t[-1],
        C_max,
        start_day=chemo_start_day,
        every=chemo_every,
        spike_duration=spike_duration,
        half_life=half_life
    )

    return t, R, G, DR, DG, DP, C


# ============================================================
# Helper to run one simulation
# ============================================================
def run_model(params):
    t, R, G, DR, DG, DP, C = simulate_fusion_model_with_decay_chemo(
        params['R0'], params['G0'], params['DR0'], params['DG0'], params['DP0'],
        params['a'], params['f'], params['K'],
        params['eta'], params['EC50'], params['C_max'],
        params['dt'], params['T'],
        params['chemo_start_day'],
        params['chemo_every'],
        params['spike_duration'],
        params['half_life']
    )
    N = R + G + DR + DG + DP
    mono = R + G
    dip = DR + DG + DP
    return t, R, G, DR, DG, DP, C, N, mono, dip


# ============================================================
# PARAMETERS
# ============================================================
params = {
    'R0': 50,
    'G0': 100,
    'DR0': 0,
    'DG0': 0,
    'DP0': 0,

    # Corrected growth parameter:
    'a': A_PER_DAY,
    'f': 134.4e-3,
    'K': 1e5,

    'eta': {
        'R': 3.0,
        'G': 3.0,
        'DR': 1.75,
        'DG': 1.75,
        'DP': 1.5
    },

    # Fixed EC50 for all populations
    'EC50': {
        'R': EC50_COMMON_uM,
        'G': EC50_COMMON_uM,
        'DR': EC50_COMMON_uM,
        'DG': EC50_COMMON_uM,
        'DP': EC50_COMMON_uM
    },

    'C_max': 6.0,

    # Treatment schedule
    'chemo_start_day': 21.0,      # first treatment at day 21
    'chemo_every': 21.0,          # every 3 weeks
    'spike_duration': DT_15_MIN,  # 15-minute administration
    'half_life': 0.5,             # 12 hours

    # Numerics
    'dt': DT_15_MIN,              # exactly 15 minutes
    'T': 120.0
}


# ============================================================
# RUN BASELINE SIMULATION
# ============================================================
t, R, G, DR, DG, DP, C, N, mono, dip = run_model(params)


# ============================================================
# BASELINE PLOTS
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(t, C, color=COLORS[3])
plt.axvline(21.0, color='black', linestyle='--', alpha=0.7, label='First treatment (day 21)')
plt.title("Chemotherapy: starts at day 21, every 3 weeks, half-life 12 hours")
plt.xlabel("Time (days)")
plt.ylabel("Drug concentration (µM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, R, label="R", color=COLORS[0])
plt.plot(t, G, label="G", color=COLORS[1])
plt.plot(t, DR, label="DR", color=COLORS[2])
plt.plot(t, DG, label="DG", color=COLORS[3])
plt.plot(t, DP, label="DP", color=COLORS[4])
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, N, label="Total population", color=COLORS[5])
plt.xlabel("Time (days)")
plt.ylabel("Total population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, mono, label="Monoploid", color=COLORS[6])
plt.plot(t, dip, label="Diploid", color=COLORS[7])
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

eps = 1e-12
plt.figure(figsize=(10, 6))
plt.plot(t, R / (N + eps), label="R fraction", color=COLORS[0])
plt.plot(t, G / (N + eps), label="G fraction", color=COLORS[1])
plt.plot(t, DR / (N + eps), label="DR fraction", color=COLORS[2])
plt.plot(t, DG / (N + eps), label="DG fraction", color=COLORS[3])
plt.plot(t, DP / (N + eps), label="DP fraction", color=COLORS[4])
plt.xlabel("Time (days)")
plt.ylabel("Fraction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# SWEEP: vary all eta_i together and vary C_max
# ============================================================
eta_base = params['eta'].copy()

eta_scale_range = np.linspace(0.2, 2.0, 30)
Cmax_range = np.linspace(0.1, 80.0, 40)   # widened because EC50 is now ~47.3 µM

final_total = np.zeros((len(eta_scale_range), len(Cmax_range)))
final_DP = np.zeros((len(eta_scale_range), len(Cmax_range)))

for i, eta_scale in enumerate(eta_scale_range):
    for j, C_test in enumerate(Cmax_range):
        test_params = params.copy()
        test_params['eta'] = {
            'R': eta_base['R'] * eta_scale,
            'G': eta_base['G'] * eta_scale,
            'DR': eta_base['DR'] * eta_scale,
            'DG': eta_base['DG'] * eta_scale,
            'DP': eta_base['DP'] * eta_scale,
        }
        test_params['C_max'] = C_test

        t2, R2, G2, DR2, DG2, DP2, C2, N2, mono2, dip2 = run_model(test_params)

        final_total[i, j] = N2[-1]
        final_DP[i, j] = DP2[-1]


# ============================================================
# Heatmap: final total population
# ============================================================
plt.figure(figsize=(9, 6))
plt.imshow(
    final_total,
    origin="lower",
    aspect="auto",
    extent=[
        Cmax_range.min(), Cmax_range.max(),
        eta_scale_range.min(), eta_scale_range.max()
    ]
)
plt.xlabel("C_max (µM)")
plt.ylabel("eta scale")
plt.title("Final total population N(T)")
plt.colorbar(label="N(T)")
plt.tight_layout()
plt.show()


# ============================================================
# Heatmap: final DP population
# ============================================================
plt.figure(figsize=(9, 6))
plt.imshow(
    final_DP,
    origin="lower",
    aspect="auto",
    extent=[
        Cmax_range.min(), Cmax_range.max(),
        eta_scale_range.min(), eta_scale_range.max()
    ]
)
plt.xlabel("C_max (µM)")
plt.ylabel("eta scale")
plt.title("Final DP population DP(T)")
plt.colorbar(label="DP(T)")
plt.tight_layout()
plt.show()


# ============================================================
# Line plots: N(T) vs C_max for selected eta scales
# ============================================================
eta_scale_selected = [0.25, 0.5, 1.0, 1.5, 2.0]

plt.figure(figsize=(9, 6))

for idx, eta_scale in enumerate(eta_scale_selected):
    N_final_curve = []

    for C_test in Cmax_range:
        test_params = params.copy()
        test_params['eta'] = {
            'R': eta_base['R'] * eta_scale,
            'G': eta_base['G'] * eta_scale,
            'DR': eta_base['DR'] * eta_scale,
            'DG': eta_base['DG'] * eta_scale,
            'DP': eta_base['DP'] * eta_scale,
        }
        test_params['C_max'] = C_test

        t2, R2, G2, DR2, DG2, DP2, C2, N2, mono2, dip2 = run_model(test_params)
        N_final_curve.append(N2[-1])

    plt.plot(Cmax_range, N_final_curve, label=f"eta scale = {eta_scale}", color=COLORS[idx % len(COLORS)])

plt.xlabel("C_max (µM)")
plt.ylabel("Final total population N(T)")
plt.title("N(T) vs C_max for different eta scales")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Line plots: DP(T) vs C_max for selected eta scales
# ============================================================
plt.figure(figsize=(9, 6))

for idx, eta_scale in enumerate(eta_scale_selected):
    DP_final_curve = []

    for C_test in Cmax_range:
        test_params = params.copy()
        test_params['eta'] = {
            'R': eta_base['R'] * eta_scale,
            'G': eta_base['G'] * eta_scale,
            'DR': eta_base['DR'] * eta_scale,
            'DG': eta_base['DG'] * eta_scale,
            'DP': eta_base['DP'] * eta_scale,
        }
        test_params['C_max'] = C_test

        t2, R2, G2, DR2, DG2, DP2, C2, N2, mono2, dip2 = run_model(test_params)
        DP_final_curve.append(DP2[-1])

    plt.plot(Cmax_range, DP_final_curve, label=f"eta scale = {eta_scale}", color=COLORS[idx % len(COLORS)])

plt.xlabel("C_max (µM)")
plt.ylabel("Final DP population DP(T)")
plt.title("DP(T) vs C_max for different eta scales")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()