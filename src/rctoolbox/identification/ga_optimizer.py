import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import warnings

# GA pymoo imports
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan

from rctoolbox.evaluation.forecast_metrics import compute_forecast_metrics

def summarize_models(model_array):
    param_names = model_array[0].keys()
    param_values = {key: [] for key in param_names}
    for model in model_array:
        for key in param_names:
            param_values[key].append(float(model[key]))
    summary = {key: (float(round(np.mean(values),2)), float(round(np.std(values),2))) for key, values in param_values.items()}
    return summary


_CRITERIA_FORMULAS = {
    "val error":         lambda df: df["validation_error"],
    "train error":       lambda df: df["train_error"],
    "train + val error": lambda df: df["train_error"] + df["validation_error"],
}


def add_criteria_column(df, criteria):
    """Return a copy of ``df`` with a ``criteria`` column derived from ``criteria`` name."""
    if criteria not in _CRITERIA_FORMULAS:
        raise ValueError(
            f"Unknown criteria '{criteria}'. Expected one of: {sorted(_CRITERIA_FORMULAS)}."
        )
    out = df.copy()
    out["criteria"] = _CRITERIA_FORMULAS[criteria](out)
    return out

class GAOptimizer:
    def __init__(self, parameter_space, train_data: dict = {}, val_data: dict = None, objective_function: callable = None):

        self.parameter_space = parameter_space
        # Backward-compat alias: existing notebook code reads optimizer.bldg_props.X
        self.bldg_props = parameter_space
        self.setup_param_bounds()

        self.train_data = train_data
        self.val_data = val_data

        self.objective_function = objective_function

    def set_model_and_problem(self, model, problem):
        """Inject the Model instance and the constructed BaseProblem."""
        self.building_thermal_model = model
        self.problem = problem

    def setup_param_bounds(self):
        self.free_param_bounds = self.parameter_space.free_param_bounds
        self.free_param_min_bounds = self.parameter_space.xl
        self.free_param_max_bounds = self.parameter_space.xu
        self.free_param_names = list(self.parameter_space.free_param_names)
        self.n_free_params = self.parameter_space.n_free_params
        # # Some 5R1C-aware code reads this directly off the optimizer; surface it
        # # transparently when the parameter space supplies it.
        # self.bldg_phys_param_bounds = getattr(
        #     self.parameter_space, "bldg_phys_param_bounds", None
        # )

    def set_optimisation_parameters(self, pop_size, n_gen):
        self.pop_size, self.n_gen = pop_size, n_gen

    def run_optimization(self, seed=42):
        algorithm = GA(pop_size=self.pop_size, eliminate_duplicates=True)
        # algorithm = GA(pop_size=self.pop_size, eliminate_duplicates=True, crossover=SBX(prob=0.9, eta=15), mutation=PM(prob=1/self.n_free_params, eta=20))
        termination = get_termination("n_gen", self.n_gen)
        self.result = minimize(self.problem, algorithm, termination, seed=seed, save_history=True, verbose=True)

    def save_results(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.result, f)

    def load_results(self, path):
        with open(path, "rb") as f:
            self.result = pickle.load(f)

    def evaluate_on_validation(self, N=None, use_train_err_threshold=True):
        R = []
        if N is None:
            N = self.n_gen
        for gen_idx, entry in enumerate(self.result.history[-N:]):
            generation_number = len(self.result.history) - N + gen_idx + 1
            X_gen = entry.pop.get("X")
            F_gen = entry.pop.get("F").flatten()
            train_error_threshold = F_gen.min() + F_gen.std() if use_train_err_threshold else np.inf
            sorted_indices = np.argsort(F_gen)

            for rank_in_gen, indiv_idx in enumerate(sorted_indices):
                x = X_gen[indiv_idx]
                train_error = round(float(F_gen[indiv_idx]), 5)
                if train_error <= train_error_threshold:
                    model_params = {k: float(v) for k, v in zip(self.free_param_names, x)}

                    y_pred = self.problem.predict(model_params, self.val_data)
                    y_true = self.problem.get_truth(self.val_data)
                    obj_val = self.objective_function(y_true, y_pred)
    
                    R.append({
                        "generation": generation_number,
                        "rank_in_generation": rank_in_gen,
                        "individual_index": indiv_idx,
                        "train_error": train_error,
                        "validation_error": round(float(obj_val), 5),
                        "params": model_params
                    })

        self.dfR = pd.DataFrame(R)

    def evaluate_on_train_only(self, N=None):
        R = []
        if N is None:
            N = self.n_gen
        for gen_idx, entry in enumerate(self.result.history[-N:]):
            generation_number = len(self.result.history) - N + gen_idx + 1
            X_gen = entry.pop.get("X")
            F_gen = entry.pop.get("F").flatten()
            sorted_indices = np.argsort(F_gen)

            for rank_in_gen, indiv_idx in enumerate(sorted_indices):
                x = X_gen[indiv_idx]
                train_error = round(float(F_gen[indiv_idx]), 5)
                model_params = {k: float(v) for k, v in zip(self.free_param_names, x)}
                R.append({
                    "generation": generation_number,
                    "rank_in_generation": rank_in_gen,
                    "individual_index": indiv_idx,
                    "train_error": train_error,
                    "validation_error": np.nan,
                    "params": model_params
                })

        self.dfR = pd.DataFrame(R)

    def compute_best_candidate(self, criteria):
        """Pick the single best model from ``dfR`` by the given criteria.

        Independent of R* selection: the argmin over the top-N slice equals
        the argmin over the full history, so this only needs ``dfR``.
        Sets ``self.best_candidate`` and adds a ``criteria`` column to
        ``self.dfR`` so downstream selection can reuse it.
        """
        self.dfR = add_criteria_column(self.dfR, criteria)

        best_idx = self.dfR["criteria"].idxmin()
        best_params = self.dfR.loc[best_idx, "params"]
        best_params = {key: round(float(val), 5) for key, val in best_params.items()}

        self.best_candidate = {
            "params": best_params,
            "train_error": self.dfR.loc[best_idx, "train_error"],
            "validation_error": self.dfR.loc[best_idx, "validation_error"],
            "criteria": self.dfR.loc[best_idx, "criteria"],
        }

    def get_final_model(self):
        """Compute distribution stats over the selected pool ``dfRtopN``."""
        self.final_models = pd.DataFrame(self.dfRtopN["params"].values.tolist())
        self.final_model_dist = summarize_models(self.dfRtopN["params"].values)
        self.final_model_average = {key: val[0] for key, val in self.final_model_dist.items()}
        
    def run_clustering_analysis(self, min_cluster_sizes=list(range(10, 60, 2))):
        warnings.filterwarnings("ignore", category=FutureWarning)

        param_vectors = np.array([list(p.values()) for p in self.dfRtopN["params"].values])
        self.param_names = list(self.dfRtopN["params"].values[0].keys())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(param_vectors)

        results = []
        best_config = None
        lowest_criteria = float("inf")
        criteria_vals = self.dfRtopN["criteria"].values

        for min_size in min_cluster_sizes:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if n_clusters > 1:
                sil_score = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
            else:
                sil_score = np.nan

            mean_criteria = np.mean(criteria_vals[np.array(labels) != -1]) if n_clusters > 0 else np.nan

            results.append({
                "min_cluster_size": min_size,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette_score": sil_score,
                "mean_criteria_non_noise": mean_criteria,
                "labels": labels 
            })

            if n_clusters >= 2 and not np.isnan(sil_score):
                if mean_criteria < lowest_criteria:
                    lowest_criteria = mean_criteria
                    best_config = {
                        "min_cluster_size": min_size,
                        "n_clusters": n_clusters,
                        "silhouette_score": sil_score,
                        "mean_criteria": mean_criteria,
                        "labels": labels
                    }

        self.clustering_results_df = pd.DataFrame(results)
        self.best_clustering_config = best_config

        if best_config is not None:
            self.dfRtopN["best_cluster"] = best_config["labels"]

    def apply_clustering_config(self, index=0):
        if not hasattr(self, 'clustering_results_df'):
            raise RuntimeError("No clustering configurations stored. Run run_clustering_analysis() first.")
        
        if index < 0 or index >= len(self.clustering_results_df):
            raise IndexError(f"Invalid index: {index}. Must be between 0 and {len(self.clustering_results_df) - 1}")
        
        self.dfRtopN["best_cluster"] = self.clustering_results_df.loc[index, "labels"]
        print(f"Applied clustering configuration of index = {index}")

    def process_identified_clusters(self):
        if "best_cluster" not in self.dfRtopN.columns:
            raise RuntimeError("No clustering has been applied.")

        cluster_summaries = {}
        best_models = {}
        best_models_list = []

        cluster_ids = np.unique(self.dfRtopN["best_cluster"].values)

        for cluster_id in cluster_ids:
            if cluster_id == -1:
                continue  # skip noise if using HDBSCAN

            cluster_summaries[int(cluster_id)] = {}

            cluster_df = self.dfRtopN[self.dfRtopN["best_cluster"] == cluster_id]

            cluster_summaries[int(cluster_id)]["count"] = len(cluster_df)

            # Select average parameter vector for the cluster and compute std
            vectors = np.array([list(p.values()) for p in cluster_df["params"].values])

            mean_vals = np.mean(vectors, axis=0)
            std_vals = np.std(vectors, axis=0)

            min_vals = np.min(vectors, axis=0)
            max_vals = np.max(vectors, axis=0)

            p05_vals = np.percentile(vectors, 5, axis=0)
            p50_vals = np.percentile(vectors, 50, axis=0)
            p95_vals = np.percentile(vectors, 95, axis=0)

            params_mean = {self.free_param_names[i]: round(float(mean_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_std = {self.free_param_names[i]: round(float(std_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_min = {self.free_param_names[i]: round(float(min_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_max = {self.free_param_names[i]: round(float(max_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_p05 = {self.free_param_names[i]: round(float(p05_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_p50 = {self.free_param_names[i]: round(float(p50_vals[i]), 2) for i in range(len(self.free_param_names))}
            params_p95 = {self.free_param_names[i]: round(float(p95_vals[i]), 2) for i in range(len(self.free_param_names))}

            cluster_summaries[int(cluster_id)]["mean"] = params_mean
            cluster_summaries[int(cluster_id)]["std"] = params_std
            cluster_summaries[int(cluster_id)]["min"] = params_min
            cluster_summaries[int(cluster_id)]["max"] = params_max
            cluster_summaries[int(cluster_id)]["p05"] = params_p05
            cluster_summaries[int(cluster_id)]["p50"] = params_p50
            cluster_summaries[int(cluster_id)]["p95"] = params_p95

            # Select candidate from cluster with lowest criteria
            best_idx = cluster_df["criteria"].idxmin()
            best_row = cluster_df.loc[best_idx]
            best_candidate_set = {key: round(float(val), 5) for key, val in best_row["params"].items()}

            cluster_summaries[int(cluster_id)]["best_candidate"] = {
                "params": best_candidate_set,
                "criteria": best_row["criteria"],
                "validation_error": best_row["validation_error"],
                "train_error": best_row["train_error"]
            }

            # Store also in list form for dataframe
            row_dict = best_candidate_set.copy()
            row_dict["cluster"] = int(cluster_id)
            row_dict["criteria"] = best_row["criteria"]
            best_models_list.append(row_dict)

        self.cluster_summaries = cluster_summaries
        self.cluster_best_models = best_models
        self.cluster_best_models_df = pd.DataFrame(best_models_list)
        self.final_models["cluster"] = self.dfRtopN["best_cluster"]

        self.cmap = plt.get_cmap("brg", len(self.cluster_summaries.keys()))
        self.cluster_to_color = {g: self.cmap(i) for i, g in enumerate(self.cluster_summaries.keys())}


    def evaluate_clusters(self, metric="MAE", criteria="val error"):

        cluster_performance = {}

        for key, params_cluster in self.cluster_summaries.items():
            if key != -1:
                representative_params_set = params_cluster["best_candidate"]["params"]

                # EVALUATE ON TRAIN
                y_pred_train = self.problem.predict(representative_params_set, self.train_data)
                y_true_train = self.problem.get_truth(self.train_data)
                train_performance = compute_forecast_metrics(y_true_train, y_pred_train)
                train_error = train_performance[metric]

                # EVALUATE ON VAL (only if val_data is available)
                val_error = None
                if self.val_data:
                    y_pred_val = self.problem.predict(representative_params_set, self.val_data)
                    y_true_val = self.problem.get_truth(self.val_data)
                    val_performance = compute_forecast_metrics(y_true_val, y_pred_val)
                    val_error = val_performance[metric]

                if criteria == "train + val error":
                    score = train_error + val_error
                elif criteria == "val error":
                    score = val_error
                elif criteria == "train error":
                    score = train_error

                cluster_performance[key] = {'cl_eval_score': score}
                self.cluster_summaries[key]["best_candidate"]["cl_eval_score"] = score

        self.cluster_performance = cluster_performance

    # def plot_clustering_metrics(self):
    #     if not hasattr(self, 'clustering_results_df'):
    #         raise RuntimeError("Run clustering analysis first.")

    #     df = self.clustering_results_df
    #     fig, axes = plt.subplots(4, 1, figsize=(5, 5), sharex=True)

    #     axes[0].plot(df["min_cluster_size"], df["silhouette_score"], marker='o')
    #     axes[0].set_ylabel("Silhouette Score")
    #     axes[0].set_title("HDBSCAN Clustering Metrics vs. min_cluster_size")
    #     axes[0].grid(True)

    #     axes[1].plot(df["min_cluster_size"], df["n_clusters"], marker='x')
    #     axes[1].set_ylabel("Number of Clusters")
    #     axes[1].grid(True)

    #     axes[2].plot(df["min_cluster_size"], df["n_noise"], marker='s')
    #     axes[2].set_ylabel("Noise Points")
    #     axes[2].grid(True)

    #     axes[3].plot(df["min_cluster_size"], df["mean_criteria_non_noise"], marker='d')
    #     axes[3].set_ylabel("Mean Criteria")
    #     axes[3].set_xlabel("min_cluster_size")
    #     axes[3].grid(True)

    #     plt.tight_layout()
    #     plt.show()