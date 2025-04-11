
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import CubicSpline
import pymatviz as pmv
from pymatgen.core import Composition
from emmet.core.neb import NebPathwayResult, NebResult

class TSAnlysis:
    def __init__(
        self,
        data: dict,
        ready_data: bool = True,
        barrier_type: str = "barrier_energy_range",
        barrier_cutoff: float = float("inf"),
    ):
        self.DFT = data["DFT"]
        self.MLIP = {k:v for k, v in data.items() if k != "DFT"}
        self.set_attr_mlip_res()
        if ready_data:
            self.ready_data(
                barrier_type=barrier_type,
                cutoff=barrier_cutoff
            )

    def set_attr_mlip_res(self):
        for mlip_method, data in self.MLIP.items():
            setattr(self, mlip_method, data)

    def add_npr_docs_from_res(self, method = "all"):
        if method == "all":
            convert_list = ["DFT"] + list(self.MLIP.keys())
        else:
            convert_list = [method]
        for attr in convert_list:
            data = getattr(self, attr)
            npr_docs = self.res_to_npr_docs(data)
            setattr(self, attr + "_npr_docs", npr_docs)

    def flatten_docs(self, method="all"):
        """flatten the npr docs and convert them into hops"""
        if method == "all":
            convert_list = ["DFT"] + list(self.MLIP.keys())
        else:
            convert_list = [method]
        for attr in convert_list:
            flattened_hops = {}
            npr_docs = getattr(self, attr + "_npr_docs")
            for battery_id, data in npr_docs.items():
                for hop_key, hop_data in data.hops.items():
                    key = f"{battery_id}.{hop_key}"
                    flattened_hops[key] = hop_data
            setattr(self, attr + "_all_hops", flattened_hops)

    def get_analysis_hops(
        self,
        barrier_type: str,
        cutoff: float
    ):
        hops_attr = [i for i in list(vars(self).keys()) if ("_all_hops" in i)]
        analysis_hops = {method: {} for method in hops_attr}
        for batt_hop_key, nr in self.DFT_all_hops.items():
            DFT_barrier = self._get_barrier_max(nr) if barrier_type == "barrier_max" else getattr(nr, barrier_type)
            if DFT_barrier <= cutoff:
                for attr in hops_attr:
                    analysis_hops[attr][batt_hop_key] = getattr(self, attr)[batt_hop_key]
        for attr in hops_attr:
            method = attr.split("_")[0]
            setattr(self, method + "_analysis_hops", analysis_hops[attr])

    def populate_analysis_data(
        self,
        barrier_type: str,
    ):
        hops_attr = [i for i in list(vars(self).keys()) if ("_analysis_hops" in i)]
        properties = [
            "analysis_barrier",
            "spline_sign_change",
            "energy_sign_change",
            "energy_above_min",
            "flattened_e_above_min",
            "energy_diff_sign"
        ]
        analysis_data = {
            property: {method: {} for method in hops_attr}
            for property in properties
        }

        for batt_hop_key, DFT_nr in self.DFT_analysis_hops.items():
            min_energy_idx = DFT_nr.energies.index(min(DFT_nr.energies))
            for attr in hops_attr:
                nr = getattr(self, attr)[batt_hop_key]
                # below are the operation to get analysis data
                # barrier
                barrier = self._get_barrier_max(nr) if barrier_type == "barrier_max" else getattr(nr, barrier_type)
                analysis_data["analysis_barrier"][attr][batt_hop_key] = barrier
                # spline fit first derivative sign change
                analysis_data["spline_sign_change"][attr][batt_hop_key] = self._get_spline_sign_change(nr)
                # sign change of derivative of energies
                analysis_data["energy_sign_change"][attr][batt_hop_key] = self._get_energy_sign_change(nr)
                # energy of each image above the min energy in hop
                energy_above_min = self._get_e_above_min(nr, min_energy_idx)
                analysis_data["energy_above_min"][attr][batt_hop_key] = energy_above_min
                for idx, energy in enumerate(energy_above_min):
                    analysis_data["flattened_e_above_min"][attr][batt_hop_key + "." + str(idx)] = energy
                # array of sign of diff in energy
                analysis_data["energy_diff_sign"][attr][batt_hop_key] = self._get_point_wise_diff_sign(nr)

        for attr in hops_attr:
            method = attr.split("_")[0]
            for data_type in analysis_data:
                setattr(self, method + "_" + data_type, analysis_data[data_type][attr])

    def ready_data(self, barrier_type, cutoff):
        self.add_npr_docs_from_res()
        self.flatten_docs()
        self.get_analysis_hops(
            barrier_type=barrier_type,
            cutoff=cutoff
        )
        self.populate_analysis_data(
            barrier_type=barrier_type
        )
        self.generate_analysis_df()

    def generate_analysis_df(
        self,
        additional_data: dict[str, dict] = None,
        operate: bool = True,
        last_active_species: str = "Li",
        randomize_rows: bool = False,
    ):
        df_keys = [
            "flattened_e_above_min",
        ]
        df_attr_names = [attr for attr in vars(self).keys() if [i for i in df_keys if i in attr]]
        df_dicts = {
            key: getattr(self, key) for key in df_attr_names
        }

        if additional_data:
            for key, data in additional_data.items():
                data = {k: v for k, v in data.items() if k in list(df_dicts.values())[0].keys()}
                additional_data[key] = data
            df_dicts.update(additional_data)

        df = pd.DataFrame(df_dicts)
        # delete rows that have -float("inf") as the DFT value since that's the min in hop
        df.replace([-float("inf")], np.nan, inplace=True)
        df.dropna(inplace=True)
        self.analysis_df = df
        if operate:
            self.operate_on_anlaysis_df(
                last_active_species=last_active_species,
                randomize_row=randomize_rows
            )

    def operate_on_anlaysis_df(
        self,
        last_active_species: str = "Li",
        randomize_row: bool = True
    ):
        self.analysis_df["active_species"] = self.analysis_df.index.to_series().apply(self.get_active_ion_from_flattened_id)
        self.analysis_df = self.analysis_df.sort_values(by="active_species", key=lambda x: x == last_active_species, ascending=True)
        if "composition" in self.analysis_df.keys():
            self.analysis_df["heaviest_element"] = self.analysis_df["composition"].apply(self._extract_heaviest_element)
            self.analysis_df["heaviest_element_bin"] = self.analysis_df["heaviest_element"].apply(self._bin_heavy_element)
        if randomize_row:
            self.analysis_df = self.analysis_df.sample(frac=1)

    def get_active_ion_from_flattened_id(
        self,
        flattened_id: str,
    ):
        npr_id = flattened_id.split(".")[0]
        return self.DFT_npr_docs[npr_id].active_species

    def get_plot_data(
        self,
        mlip_method: str,
        type: str
    ):
        plot_data = {k: [] for k in ["DFT", "MLIP"]}
        attr_name = "_" + type

        for batt_hop_key, DFT_data in getattr(self, "DFT" + attr_name).items():
            plot_data["DFT"].append(DFT_data)
            plot_data["MLIP"].append(getattr(self, mlip_method + attr_name)[batt_hop_key])
        return plot_data

    def get_hued_parity_plot_from_df(
        self,
        mlip_method: str,
        hue: str = "active_species",
        **kwargs
    ):
        df = self.analysis_df
        mlip_column_name = mlip_method + "_flattened_e_above_min"

        fig, ax = plt.subplots(figsize=kwargs["figsize"] if "figsize" in kwargs else (8, 6))
        sns.scatterplot(
            x='DFT_flattened_e_above_min',
            y=mlip_column_name,
            hue=hue,
            data=df,
            palette='colorblind',
            s=kwargs["s"] if "s" in kwargs else 25,
            linewidth=0,
            alpha=kwargs["alpha"] if "alpha" in kwargs else 0.5,
            ax=ax
        )

        max_val = 5.25
        ax.plot([0, max_val], [0, max_val], linestyle='--', color='black')

        ax.set_xlim(None, max_val)
        ax.set_xlabel("DFT-calculated E above min")
        ax.set_ylabel(f"{mlip_method}-static E above min")
        ax.set_title(f"{mlip_method} Energy Benchmark")
        ax.legend(title=hue)

        return ax

    def get_barrier_scatter_plot(
        self,
        mlip_method: str,
        **kwargs
    ):
        plot_data = self.get_plot_data(mlip_method, "analysis_barrier")
        x_label = "DFT-calculated barriers (eV)"
        y_label = f"{mlip_method}-static barriers (eV)"
        title = f"{mlip_method} Barrier Benchmark"
        return self.plot_density_scatter(
            plot_data=plot_data,
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            **kwargs
        )

    def get_sign_change_plot(
        self,
        mlip_method: str,
        type = "energy_diff_sign",
        **kwargs
    ):
        """type can be energy_sign_change, spline_energy_change, or energy_diff_sign"""
        plot_data = self.get_plot_data(mlip_method, type)
        if type == "energy_diff_sign":
            diff_distribution = [
                sum([plot_data["MLIP"][i][j] != plot_data["DFT"][i][j] for j in range(len(plot_data["DFT"][i]))])
                for i in range(len(plot_data["DFT"]))
            ]
        else:
            diff_distribution = [
                plot_data["MLIP"][i] - plot_data["DFT"][i]
                for i in range(len(plot_data["DFT"]))
            ]

        x = list(set(diff_distribution))
        height = [diff_distribution.count(value) for value in x]
        plt.xlabel("Error in TS shape")
        plt.ylabel("Number of migration events")
        plt.title(f"{mlip_method} TS Shape Benchmark")
        return plt.bar(
            x,
            height,
        )

    def get_point_wise_energy_error_plot(
        self,
        mlip_method: str,
        **kwargs
    ):
        plot_data = self.get_plot_data(mlip_method, "flattened_e_above_min")
        # delete the minimum energy points which are assigned -inf at creation
        for k, v in plot_data.items():
            plot_data[k] = [i for i in v if i != -float("inf")]
        x_label = "DFT-calculated energy above minimum (eV)"
        y_label = f"{mlip_method} energy above min (eV)"
        title = f"{mlip_method} Point-Wise Energy Benchmark"
        return self.plot_density_scatter(
            plot_data=plot_data,
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            **kwargs
        )


    @staticmethod
    def plot_density_scatter(
        plot_data,
        title=None,
        **kwargs
    ):
        x = np.array(plot_data["DFT"])
        y = np.array(plot_data["MLIP"])
        plt.title(title)
        return pmv.density_scatter(
            x=x,
            y=y,
            color_bar=False,
            cmap="viridis",
            **kwargs
        )

    @staticmethod
    def _get_barrier_max(nr: NebResult):
        return max(nr.forward_barrier, nr.reverse_barrier)

    @staticmethod
    def _get_spline_sign_change(nr: NebResult):
        ba = nr.barrier_analysis
        spline_fit = CubicSpline(
            ba.frame_index,
            ba.energies,
            bc_type="clamped"
        )
        x_fine = np.linspace(ba.frame_index[0], ba.frame_index[-1], 1000)
        dy_dx = spline_fit.derivative()(x_fine)[1:-1] # delete end points to get correct sign change
        sign_change = np.sum(np.diff(np.sign(dy_dx)) != 0)
        return int(sign_change)

    @staticmethod
    def _get_energy_sign_change(nr: NebResult):
        energies = nr.energies
        sign_of_diff = np.sign(np.diff(energies))
        sign_change = np.sum(np.diff(sign_of_diff) != 0)
        return sign_change

    @staticmethod
    def _get_e_above_min(nr: NebResult, min_e_idx: int):
        min_energy = nr.energies[min_e_idx]
        e_above_min = []
        for idx, energy in enumerate(nr.energies):
            e_above_min.append(energy - min_energy if idx != min_e_idx else -float("inf"))
        return e_above_min

    @staticmethod
    def _get_point_wise_diff_sign(nr: NebResult):
        return np.sign(np.diff(nr.energies))

    @staticmethod
    def _extract_heaviest_element(comp: Composition):
        return max([i.Z for i in comp.elements])

    @staticmethod
    def _bin_heavy_element(Z: int, bin_width=40, hard_cutoff=100):
        bin = Z // bin_width + 1
        return f"under {min(hard_cutoff, bin*bin_width)}"

    @staticmethod
    def res_to_npr_docs(res_dict):
        npr_docs = {}
        for battery_id, data in res_dict.items():
            for species in ["Li", "Ca", "Zn", "Mg"]:
                if species in battery_id:
                    active_speices = species
            npr_forwards_barriers = {}
            npr_reverse_barriers = {}
            npr_hops = {}

            for hop_key, energies in data.items():
                nr = NebResult()
                nr.energies = energies
                nr.set_barriers()

                npr_hops[hop_key] = nr
                npr_forwards_barriers[hop_key] = nr.forward_barrier
                npr_reverse_barriers[hop_key] = nr.reverse_barrier

            npr_docs[battery_id] = NebPathwayResult(
                identifier=battery_id,
                hops=npr_hops,
                active_species=active_speices,
                forward_barriers=npr_forwards_barriers,
                reverse_barriers=npr_reverse_barriers,
            )

        return npr_docs
