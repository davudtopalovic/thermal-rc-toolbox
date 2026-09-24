# Imports
import math
import os
import json
import pickle

from rctoolbox.utils.logging import setup_logger
from rctoolbox.utils.set_global_seed import set_global_seed
from rctoolbox.utils.config_loader import load_json_config
from rctoolbox.identification.selection import select_best_solutions

_STAGE_NAMES = {1: "optimize", 2: "select", 3: "cluster"}


def parse_stage_spec(spec):
    """Parse a --stage spec into (start, end) with 1 ≤ start ≤ end ≤ 3.

    Accepts a single stage (e.g. "2") or an inclusive range (e.g. "1-3").
    Stages: 1=optimize, 2=select, 3=cluster.
    """
    s = str(spec).strip()
    try:
        if "-" in s:
            start_s, end_s = s.split("-", 1)
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(s)
    except ValueError:
        raise ValueError(
            f"Invalid --stage '{spec}'. Expected N or N-M with N,M in {{1,2,3}}."
        )
    if not (1 <= start <= end <= 3):
        raise ValueError(
            f"Invalid --stage '{spec}': must satisfy 1 ≤ start ≤ end ≤ 3."
        )
    return start, end


def run_pipeline(args, problem_kwargs=None):

    # ---------------------------
    # Logger init
    # ---------------------------
    logger = setup_logger(args.results_directory_path)

    logger.info("🚀 Starting GA optimization pipeline")
    # logger.info(f"Building: {args.building_config['hems_id']}")
    logger.info(f"Seed: {args.seed}")

    optimizer_path = os.path.join(args.results_directory_path, "optimizer.pkl")

    # ---------------------------
    # Load configs
    # ---------------------------
    logger.info("Loading building configuration...")
    if os.path.exists(args.building_config):
        building_config = load_json_config(args.building_config, "building config")
    else:
        from rctoolbox.utils.config_loader import load_default_building_config
        building_config = load_default_building_config(args.building_config)

    logger.info("Loading identification configuration...")
    config = load_json_config(args.identification_config, "identification config")

    model_config        = config.get("model_config")
    optimization_config = config.get("optimization_config")
    selection_config    = config.get("selection_config")
    clustering_config   = config.get("clustering_config")

    logger.info("Configurations loaded successfully")

    fig_directory_path = os.path.join(args.results_directory_path)
    os.makedirs(fig_directory_path, exist_ok=True)
    logger.info(f"Results directory created at: {fig_directory_path}")

    fig_directory_path = None
    if args.save_figures:
        fig_directory_path = os.path.join(args.results_directory_path, "figs")
        os.makedirs(fig_directory_path, exist_ok=True)
        logger.info(f"Figure directory created at: {fig_directory_path}")

    # ---------------------------
    # Seed
    # ---------------------------
    set_global_seed(args.seed)

    # ---------------------------
    # Load data
    # ---------------------------
    logger.info("Loading data...")
    import joblib
    data = joblib.load(args.data_path)
    train_data, val_data = data["train"], data.get("val", None)
    logger.info(f"Data loaded from: {args.data_path}")

    # ---------------------------
    # Model plugin (registry-driven)
    # ---------------------------
    from rctoolbox.pipeline.registry import get_plugin
    model_type = config["model_config"]["model_type"]
    plugin = get_plugin(model_type)
    logger.info(f"Using model plugin: {model_type}")

    parameter_space = plugin.parameter_space_cls.from_building_config(building_config)
    model = plugin.model_factory(building_config, model_config)

    # ---------------------------
    # Objective function
    # ---------------------------
    # optimization_config["objective_function"] is either a metric name
    # ("MAE", ...; the built-in objective, with std_weight) or a
    # "file.py:function" / "package.module:function" reference to a custom
    # callable. --objective-function on the command line overrides it and is
    # written back into the config so that results.json documents the
    # objective the run actually used.
    from rctoolbox.identification.objective import resolve_objective
    cli_spec = getattr(args, "objective_function", None)
    if cli_spec:
        optimization_config["objective_function"] = cli_spec
    spec = optimization_config.get("objective_function")
    if not spec:
        raise ValueError(
            "No objective configured: set optimization_config['objective_function'] "
            "to a metric name (e.g. 'MAE') or to 'file.py:function' / "
            "'package.module:function', or pass --objective-function."
        )
    objective_fn = resolve_objective(
        spec,
        std_weight=optimization_config.get("std_weight"),
        base_dir=os.path.dirname(os.path.abspath(args.identification_config)),
    )
    logger.info(f"Objective function {spec!r}: {objective_fn!r}")

    # ---------------------------
    # Optimizer + Problem
    # ---------------------------
    logger.info("Initializing GA optimizer...")
    from rctoolbox.identification.ga_optimizer import GAOptimizer

    optimizer = GAOptimizer(parameter_space, train_data, val_data, objective_function=objective_fn)

    # Merge problem kwargs from two sources:
    #   1. Caller-supplied `problem_kwargs` (takes precedence).
    #   2. `model_config["output_states"]` from the identification config JSON
    #      (handy for TemperatureProblem multi-state setups).
    pkw = dict(problem_kwargs or {})
    if "output_states" in model_config and "output_states" not in pkw:
        pkw["output_states"] = [tuple(x) for x in model_config["output_states"]]

    problem = plugin.problem_cls(
        model=model,
        parameter_space=parameter_space,
        data=train_data,
        objective_function=objective_fn,
        n_constr=plugin.problem_cls.N_CONSTR,
        **pkw,
    )

    pop_size = optimization_config["pop_size"]
    n_gen = optimization_config["n_gen"]

    logger.info(f"GA parameters → pop_size: {pop_size}, n_gen: {n_gen}")

    optimizer.set_optimisation_parameters(pop_size=pop_size, n_gen=n_gen)
    optimizer.set_model_and_problem(model, problem)

    # ---------------------------
    # Decide which stages run
    # ---------------------------
    # --stage is a range 'N' or 'N-M' with 1=optimize, 2=select, 3=cluster.
    # If the range does not start at 1, optimizer.pkl is loaded from disk to
    # supply the state produced by the skipped earlier stages.
    start, end = parse_stage_spec(args.stage)
    stages_to_run = set(range(start, end + 1))

    logger.info(
        f"Stages to run: {[_STAGE_NAMES[i] for i in sorted(stages_to_run)]}"
    )

    # Stage 3 depends on R* (dfRtopN). If stage 2 will run in this invocation
    # but is configured to skip R* selection, that combination is incoherent.
    select_R_star = selection_config.get("select_R_star", True)
    if {2, 3}.issubset(stages_to_run) and not select_R_star:
        raise ValueError(
            "Stage 3 (cluster) requires R* selection, but "
            "selection_config.select_R_star is false. Either drop stage 3 "
            "from the range or set select_R_star to true."
        )


    if 1 in stages_to_run:
        # ---------------------------
        # Run optimization
        # ---------------------------
        logger.info("========== STAGE 1: OPTIMIZE ==========")
        logger.info("Running optimization...")
        optimizer.run_optimization(seed=args.seed)
        logger.info("Optimization finished")


        if args.save_figures:
            logger.info("Saving GA evolution plots...")
            from rctoolbox.evaluation.plotting import plot_free_param_evolution
            evolution_fig_path = os.path.join(fig_directory_path, "evolution.png") if fig_directory_path else None
            plot_free_param_evolution(
                optimizer,
                show=False,
                fig_path=evolution_fig_path,
            )
            if evolution_fig_path:
                logger.info(f"Saved figure at path: {evolution_fig_path}")

        # ----------------------------------
        # Validation / train-only evaluation
        # ---------------------------------
        if val_data is not None:
            logger.info("Evaluating on validation data...")
            optimizer.evaluate_on_validation(use_train_err_threshold=False)
        else:
            logger.info("No validation data — building dfR from train history...")
            optimizer.evaluate_on_train_only()

        # Save optimizer
        if args.save_optimizer:
            with open(optimizer_path, "wb") as f:
                pickle.dump(optimizer, f)
            logger.info(f"Optimizer saved to: {optimizer_path}")

    else:
        # ---------------------------
        # Load optimizer from disk (skipping earlier stages)
        # ---------------------------
        logger.info(f"Loading optimizer from disk: {optimizer_path}")
        with open(optimizer_path, "rb") as f:
            optimizer = pickle.load(f)
        logger.info("Optimizer loaded successfully")

        if 2 in stages_to_run:
            # Selection needs dfR; recompute it if the loaded optimizer lacks it.
            if not hasattr(optimizer, "dfR"):
                if optimizer.val_data is not None:
                    logger.info("dfR missing → recomputing validation...")
                    optimizer.evaluate_on_validation(use_train_err_threshold=False)
                else:
                    logger.info("dfR missing → rebuilding from train history...")
                    optimizer.evaluate_on_train_only()
        else:
            # Starting straight at clustering requires dfRtopN on disk.
            if not hasattr(optimizer, "dfRtopN"):
                raise RuntimeError(
                    "Stage 3 (cluster) requires an optimizer with `dfRtopN` "
                    "(run stage 2 first)."
                )

    # ---------------------------
    # Selection
    # ---------------------------
    # Stage 2 always picks best_candidate by the configured criteria (over
    # the full dfR). R* cutting (dfRtopN + distribution stats) is optional,
    # controlled by selection_config.select_R_star (default true).
    if 2 in stages_to_run:
        logger.info("========== STAGE 2: SELECT ==========")
        criteria = selection_config["criteria"]

        # Re-running stage 2 invalidates any R*/cluster state from a prior
        # run stored on the loaded optimizer. Clear it before we (maybe)
        # rebuild it below.
        for attr in ("dfRtopN", "final_models", "final_model_dist",
                     "final_model_average", "cluster_summaries"):
            if hasattr(optimizer, attr):
                delattr(optimizer, attr)

        logger.info(f"Picking best candidate by criteria '{criteria}'...")
        optimizer.compute_best_candidate(criteria)

        if select_R_star:
            logger.info("Cutting R* from dfR...")
            threshold_config = selection_config["threshold_config"]
            dfRtopN, threshold, threshold_idx = select_best_solutions(
                optimizer.dfR,
                threshold_config=threshold_config,
            )
            optimizer.dfRtopN = dfRtopN
            logger.info(f"Selected {len(optimizer.dfRtopN)} top candidates")

            if args.save_figures:
                from rctoolbox.evaluation.plotting import plot_criteria_distribution_and_selection
                selection_fig_path = os.path.join(fig_directory_path, "final_space_selection.png") if fig_directory_path else None
                plot_criteria_distribution_and_selection(
                    optimizer.dfR,
                    dfRtopN,
                    criteria=criteria,
                    threshold=threshold,
                    threshold_idx=threshold_idx,
                    threshold_config=threshold_config,
                    show=False,
                    fig_path=selection_fig_path,
                )
                if selection_fig_path:
                    logger.info(f"Saved figure at path: {selection_fig_path}")

            logger.info("Computing distribution stats over R*...")
            optimizer.get_final_model()
        else:
            logger.info("select_R_star is false — skipping R* cutting.")


    # ---------------------------
    # Clustering
    # ---------------------------
    if 3 in stages_to_run:
        logger.info("========== STAGE 3: CLUSTER ==========")
        logger.info("Running clustering analysis...")

        x = len(optimizer.dfRtopN) * clustering_config["min_samples_ratio"]
        from rctoolbox.utils.get_scale import get_scale
        x = int(round(x / get_scale(x)) * get_scale(x))

        try:
            optimizer.run_clustering_analysis(min_cluster_sizes=[x])
            optimizer.apply_clustering_config(0)
        except Exception:
            optimizer.dfRtopN["best_cluster"] = -1  # ensure column exists

        # Fallback: no valid clusters → whole dfRtopN is one cluster
        if "best_cluster" not in optimizer.dfRtopN.columns or (optimizer.dfRtopN["best_cluster"] == -1).all():
            logger.info("No clusters found — treating full dfRtopN as single cluster (label=0).")
            optimizer.dfRtopN["best_cluster"] = 0

        optimizer.process_identified_clusters()

        logger.info(f"Clusters found: {len(optimizer.cluster_summaries)}")

        if args.save_figures:
            from rctoolbox.evaluation.plotting import plot_distribution_and_cluster_ranges
            cluster_ranges_fig_path = os.path.join(fig_directory_path, "distribution_and_cluster_ranges.png") if fig_directory_path else None
            plot_distribution_and_cluster_ranges(optimizer, show=False, fig_path=cluster_ranges_fig_path)
            if cluster_ranges_fig_path:
                logger.info(f"Saved figure at path: {cluster_ranges_fig_path}")

        # ---------------------------
        # Cluster evaluation
        # ---------------------------
        optimizer.evaluate_clusters(
            metric=clustering_config["cl_eval_metric"],
            criteria=clustering_config["cl_eval_criteria"]
        )

        if args.save_figures:
            from rctoolbox.evaluation.plotting import plot_cluster_performance
            cluster_perf_fig_path = os.path.join(fig_directory_path, "cluster_performance.png") if fig_directory_path else None
            plot_cluster_performance(optimizer, show=False, fig_path=cluster_perf_fig_path)
            if cluster_perf_fig_path:
                logger.info(f"Saved figure at path: {cluster_perf_fig_path}")


        # ---------------------------
        # Heat map and PCA analysis of final models
        # ---------------------------
        from rctoolbox.evaluation.plotting import plot_free_param_heatmap, plot_PCA_of_final_models
        if args.save_figures:
            heatmap_fig_path = os.path.join(fig_directory_path, "final_space_heatmap.png") if fig_directory_path else None
            plot_free_param_heatmap(optimizer, show=False, fig_path=heatmap_fig_path)
            if heatmap_fig_path:
                logger.info(f"Saved figure at path: {heatmap_fig_path}")

            pca_fig_path = os.path.join(fig_directory_path, "final_space_pca.png") if fig_directory_path else None
            plot_PCA_of_final_models(optimizer, show=False, fig_path=pca_fig_path)
            if pca_fig_path:
                logger.info(f"Saved figure at path: {pca_fig_path}")


    # ---------------------------
    # Save results
    # ---------------------------
    if stages_to_run & {2, 3}:
        if args.save_optimizer:
            with open(optimizer_path, "wb") as f:
                pickle.dump(optimizer, f)
            logger.info(f"Final Optimizer saved to: {optimizer_path}")

        logger.info("Saving results...")

        results = {
            "setup": {
                "building_config": building_config,
                "seed": args.seed,
                "identification_config": config
            },
            "results": {
                "best_candidate": optimizer.best_candidate,
                "clustering-results": getattr(optimizer, "cluster_summaries", None)
            }
        }

        def _json_safe(obj):
            if isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            return obj

        results_path = f"{args.results_directory_path}/results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, default=_json_safe)
        logger.info(f"Results saved at path: {results_path}")


        if args.save_dfRtopN and hasattr(optimizer, "dfRtopN"):
            optimizer.dfRtopN.to_csv(f"{args.results_directory_path}/dfRtopN.csv", index=False)
            logger.info(f"dfRtopN saved to: {args.results_directory_path}/dfRtopN.csv")

        logger.info("✅ Pipeline finished successfully")