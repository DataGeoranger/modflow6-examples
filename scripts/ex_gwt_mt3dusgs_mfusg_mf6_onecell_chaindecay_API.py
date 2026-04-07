from pathlib import Path

import flopy
import matplotlib.pyplot as plt
from matplotlib import get_backend
import modflowapi
import numpy as np
from flopy.plot.styles import styles

try:
    from modflow_devtools.misc import get_env, timed
except:
    import os
    from time import perf_counter

    def get_env(name, default):
        value = os.environ.get(name)
        if value is None:
            return default
        if isinstance(default, bool):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return value

    def timed(func):
        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                print(f"{func.__name__}: {perf_counter() - start:.2f}s")

        return wrapper


example_name = "ex-gwt-mf6-onecell-chaindecay-api"

try:
    import git

    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = Path(__file__).resolve().parents[1]

workspace = root / "examples"
figs_path = root / "figures"
bin_path = root / "bin"

write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)

simulation_time = 500.0
flow_nstp = int(simulation_time)
porosity = 0.2
top = 10.0
botm = 0.0
initial_head = 10.0
hydraulic_conductivity = 1.0
specific_storage = 1.0e-5
flow_hclose = 1.0e-8
flow_rclose = 1.0e-8
transport_cclose = 1.0e-6
water_injection_rate = 1.0
et_rate = water_injection_rate / (10.0 * 10.0)

pce_initial_concentration = 1000.0
tce_initial_concentration = 100.0
dce_initial_concentration = 10.0

days_per_year = 365.25
pce_decay_rate = 0.4 / days_per_year
tce_decay_rate = 0.15 / days_per_year
dce_decay_rate = 0.1 / days_per_year


def get_exe(name):
    return str(bin_path / f"{name}.exe")


def get_mf6_lib():
    candidates = [
        root / "lib" / "libmf6.dll",
        bin_path / "libmf6.dll",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("Could not find libmf6.dll for MODFLOW API run.")


def sim_root():
    return workspace / example_name


def build_mf6_models(model_ws: Path):
    model_ws.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(
        sim_name=model_ws.name,
        sim_ws=model_ws,
        exe_name=get_exe("mf6"),
    )
    flopy.mf6.ModflowTdis(
        sim,
        nper=1,
        perioddata=[(simulation_time, flow_nstp, 1.0)],
        time_units="days",
    )

    ims_flow = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=flow_hclose,
        outer_maximum=100,
        inner_maximum=100,
        inner_dvclose=flow_hclose,
        rcloserecord=flow_rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=1.0,
        filename="gwf.ims",
    )
    ims_gwt = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=transport_cclose,
        outer_maximum=100,
        inner_maximum=100,
        inner_dvclose=transport_cclose,
        rcloserecord=transport_cclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=1.0,
        filename="gwt.ims",
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True, model_nam_file="gwf.nam")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1, delr=10.0, delc=10.0, top=top, botm=botm)
    flopy.mf6.ModflowGwfic(gwf, strt=np.array([[[initial_head]]], dtype=float))
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity,
        save_specific_discharge=True,
    )
    flopy.mf6.ModflowGwfsto(gwf, ss=specific_storage, sy=0.0, transient={0: True})
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={0: [[(0, 0, 0), water_injection_rate]]}, pname="WEL-1")
    flopy.mf6.ModflowGwfevta(
        gwf,
        surface=np.array([[[initial_head]]], dtype=float),
        rate=np.array([[[et_rate]]], dtype=float),
        depth=np.array([[[1.0]]], dtype=float),
        pname="EVT-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="gwf.hds",
        budget_filerecord="gwf.bud",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.register_ims_package(ims_flow, [gwf.name])

    gwt_names = []
    for name, initial_conc, decay_rate in (
        ("gwt-pce", pce_initial_concentration, pce_decay_rate),
        ("gwt-tce", tce_initial_concentration, tce_decay_rate),
        ("gwt-dce", dce_initial_concentration, dce_decay_rate),
    ):
        gwt = flopy.mf6.ModflowGwt(sim, modelname=name, model_nam_file=f"{name}.nam", save_flows=True)
        flopy.mf6.ModflowGwtdis(gwt, nlay=1, nrow=1, ncol=1, delr=10.0, delc=10.0, top=top, botm=botm)
        flopy.mf6.ModflowGwtic(gwt, strt=np.array([[[initial_conc]]], dtype=float))
        flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, first_order_decay=True, decay=decay_rate)
        flopy.mf6.ModflowGwtssm(gwt, sources=None)
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{name}.cbc",
            concentration_filerecord=f"{name}.ucn",
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
            printrecord=[("CONCENTRATION", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{name}.gwfgwt",
        )
        gwt_names.append(gwt.name)

    for gwt_name in gwt_names:
        sim.register_ims_package(ims_gwt, [gwt_name])
    return sim


def write_models(models):
    standard_sim, api_sim = models
    standard_sim.write_simulation()
    api_sim.write_simulation()


def run_mf6_via_api(sim):
    sim_ws = Path(sim.sim_path)
    mf6 = modflowapi.ModflowApi(get_mf6_lib(), working_directory=str(sim_ws))
    mf6.set_int("ISTDOUTTOFILE", 0)
    mf6.initialize(str(sim_ws / "mfsim.nam"))
    current_time = mf6.get_current_time()
    end_time = mf6.get_end_time()
    while current_time < end_time:
        mf6.update()
        current_time = mf6.get_current_time()
    mf6.finalize()


@timed
def run_models(models, silent=True):
    standard_sim, api_sim = models
    success, buff = standard_sim.run_simulation(silent=silent, report=True)
    assert success, buff
    run_mf6_via_api(api_sim)


def load_mf6_conc(folder, modelname):
    ucn = flopy.utils.HeadFile(str(folder / f"{modelname}.ucn"), text="CONCENTRATION")
    times = np.array(ucn.get_times())
    vals = np.array([ucn.get_data(totim=t)[0, 0, 0] for t in times])
    return times, vals


def bateman_chain_solution(time):
    p0 = pce_initial_concentration
    t0 = tce_initial_concentration
    d0 = dce_initial_concentration
    l1 = pce_decay_rate
    l2 = tce_decay_rate
    l3 = dce_decay_rate
    pce = p0 * np.exp(-l1 * time)
    tce = t0 * np.exp(-l2 * time) + p0 * (l1 / (l2 - l1)) * (np.exp(-l1 * time) - np.exp(-l2 * time))
    dce = (
        d0 * np.exp(-l3 * time)
        + t0 * (l2 / (l3 - l2)) * (np.exp(-l2 * time) - np.exp(-l3 * time))
        + p0
        * l1
        * l2
        * (
            np.exp(-l1 * time) / ((l2 - l1) * (l3 - l1))
            + np.exp(-l2 * time) / ((l1 - l2) * (l3 - l2))
            + np.exp(-l3 * time) / ((l1 - l3) * (l2 - l3))
        )
    )
    return pce, tce, dce


def plot_results():
    standard_ws = sim_root() / "mf6_standard"
    api_ws = sim_root() / "mf6_api"
    std_time, std_pce = load_mf6_conc(standard_ws, "gwt-pce")
    _, std_tce = load_mf6_conc(standard_ws, "gwt-tce")
    _, std_dce = load_mf6_conc(standard_ws, "gwt-dce")
    api_time, api_pce = load_mf6_conc(api_ws, "gwt-pce")
    _, api_tce = load_mf6_conc(api_ws, "gwt-tce")
    _, api_dce = load_mf6_conc(api_ws, "gwt-dce")
    bat_pce, bat_tce, bat_dce = bateman_chain_solution(std_time)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    axes[0].plot(std_time, std_pce, color="tab:blue", lw=2.0, label="PCE (MF6 standard)")
    axes[0].plot(std_time, std_tce, color="tab:orange", lw=2.0, label="TCE (MF6 standard)")
    axes[0].plot(std_time, std_dce, color="tab:green", lw=2.0, label="DCE (MF6 standard)")
    axes[0].plot(api_time, api_pce, color="tab:blue", ls=":", lw=1.8, label="PCE (MF6 API)")
    axes[0].plot(api_time, api_tce, color="tab:orange", ls=":", lw=1.8, label="TCE (MF6 API)")
    axes[0].plot(api_time, api_dce, color="tab:green", ls=":", lw=1.8, label="DCE (MF6 API)")
    axes[0].plot(std_time, bat_pce, color="tab:blue", ls="--", lw=1.2, label="PCE (Bateman)")
    axes[0].plot(std_time, bat_tce, color="tab:orange", ls="--", lw=1.2, label="TCE (Bateman)")
    axes[0].plot(std_time, bat_dce, color="tab:green", ls="--", lw=1.2, label="DCE (Bateman)")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Concentration (mg/m3)")
    axes[0].set_title("Concentration")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(std_time, std_pce - bat_pce, color="tab:blue", lw=2.0, label="PCE (MF6 standard)")
    axes[1].plot(std_time, std_tce - bat_tce, color="tab:orange", lw=2.0, label="TCE (MF6 standard)")
    axes[1].plot(std_time, std_dce - bat_dce, color="tab:green", lw=2.0, label="DCE (MF6 standard)")
    axes[1].plot(api_time, api_pce - bat_pce, color="tab:blue", ls=":", lw=1.8, label="PCE (MF6 API)")
    axes[1].plot(api_time, api_tce - bat_tce, color="tab:orange", ls=":", lw=1.8, label="TCE (MF6 API)")
    axes[1].plot(api_time, api_dce - bat_dce, color="tab:green", ls=":", lw=1.8, label="DCE (MF6 API)")
    axes[1].axhline(0.0, color="0.2", lw=1.0, ls=":")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Model - Bateman chain (mg/m3)")
    axes[1].set_title("Residual Vs Bateman")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    fig.suptitle("MF6 One-Cell Comparison: Standard MF6, API MF6, and Bateman Chain")
    if plot_show and "agg" not in get_backend().lower():
        plt.show()
    if plot_save:
        figs_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs_path / f"{example_name}.png", dpi=300)


def scenario(silent=True):
    models = (
        build_mf6_models(sim_root() / "mf6_standard"),
        build_mf6_models(sim_root() / "mf6_api"),
    )
    if write:
        write_models(models)
    if run:
        run_models(models, silent=silent)


scenario()

if plot:
    with styles.USGSMap():
        plot_results()
