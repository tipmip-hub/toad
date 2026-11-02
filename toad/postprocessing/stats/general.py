import logging
from typing import Callable, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import inconsistent, linkage
from scipy.spatial.distance import squareform

# Dictionary mapping score names to their method names
score_dictionary = {
    "heaviside": "score_heaviside",
    "consistency": "score_consistency",
    "spatial_autocorrelation": "score_spatial_autocorrelation",
    "nonlinearity": "score_nonlinearity",
}

logger = logging.getLogger("TOAD")


class GeneralStats:
    """General cluster statistics, such as cluster score."""

    def __init__(self, toad, var: str):
        """Initialize GeneralStats.

        Args:
            toad: TOAD object
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    @overload
    def score_heaviside(
        self,
        cluster_id: int,
        return_score_fit: Literal[False] = False,
        aggregation: Literal["mean", "sum", "std", "median", "percentile", "max", "min"]
        | str = "mean",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
    ) -> float: ...

    @overload
    def score_heaviside(
        self,
        cluster_id: int,
        return_score_fit: Literal[True],
        aggregation: Literal["mean", "sum", "std", "median", "percentile", "max", "min"]
        | str = "mean",
        percentile: Optional[float] = None,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
    ) -> Tuple[float, np.ndarray]: ...

    def score_heaviside(
        self,
        cluster_id,
        return_score_fit=False,
        aggregation: Literal["mean", "sum", "std", "median", "percentile", "max", "min"]
        | str = "mean",
        percentile=None,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
    ):
        """Evaluates how closely the spatially aggregated cluster time series resembles a perfect Heaviside function.

        A score of 1 indicates a perfect step function, while 0 indicates a linear trend.

        Args:
            cluster_id: ID of the cluster to score.
            return_score_fit: If True, returns linear regression fit along with score.
            aggregation: How to aggregate spatial data. Options are:
                - "mean" - Average across space
                - "median" - Median across space
                - "sum" - Sum across space
                - "std" - Standard deviation across space
                - "percentile" - Percentile across space (requires percentile arg)
                - "max" - Maximum across space
                - "min" - Minimum across space
            percentile: Percentile value between 0-1 when using percentile aggregation.
            normalize: How to normalize the data. Options are:
                - "max" - Normalize by maximum value
                - "max_each" - Normalize each trajectory by its own maximum value
                - None: Do not normalize

        Returns:
            float: Cluster score between 0-1 if return_score_fit is False.
            tuple: (score, linear_fit) if return_score_fit is True, where score is a float between 0-1 and linear_fit is the fitted values.

        References:
            Kobe De Maeyer Master Thesis (2025)
        """

        # Does not work with raw
        if aggregation == "raw":
            raise ValueError("raw is not supported for this method.")

        # Get the variable values
        xvals = self.td.numeric_time_values  # use numeric time values for fitting
        yvals = self.td.get_timeseries(
            self.var,
            cluster_id=cluster_id,
            aggregation=aggregation,
            percentile=percentile,
            normalize=normalize,
        ).values

        # Drop NaN/Inf to avoid silent NaN propagation (common in satellite/observational data)
        valid = np.isfinite(xvals) & np.isfinite(yvals)
        x_valid = xvals[valid]
        y_valid = yvals[valid]

        if len(x_valid) < 2:
            # Not enough points for linear fit (polyfit returns empty residuals for ≤2 points)
            standardized_score = 0.0
            _score_fit = np.full_like(xvals, np.nan)
        else:
            # Perform linear regression on valid data only
            a, b = np.polyfit(x_valid, y_valid, 1)
            # Compute RSS manually (polyfit with full=True returns empty residuals for ≤2 points)
            fitted = np.polyval([a, b], x_valid)
            rss = float(np.sum((y_valid - fitted) ** 2))

            # Compute theoretical maximum RSS for a perfect Heaviside function
            heaviside_rss = float(np.sum((y_valid - np.median(y_valid)) ** 2))
            standardized_score = float(rss / heaviside_rss if heaviside_rss > 0 else 0)
            # Linear fit over full time range (for optional return)
            _score_fit = b + a * xvals

        if return_score_fit:
            return standardized_score, _score_fit
        else:
            return standardized_score

    def score_consistency(
        self,
        cluster_id: int,
    ) -> float:
        """Measures how internally consistent a cluster is by analyzing the similarity between its time series.

        Uses hierarchical clustering to group similar time series and computes an inconsistency score.
        The final score is inverted so higher values indicate more consistency.

        The method works by:
        1. Computing pairwise R² correlations between all time series in the cluster
        2. Converting correlations to distances (1 - R²)
        3. Performing hierarchical clustering using Ward linkage
        4. Calculating inconsistency coefficients at the highest level
        5. Converting to a consistency score by taking the inverse

        Args:
            cluster_id: ID of the cluster to evaluate.

        Returns:
            float: Consistency score between 0-1, where:
                1.0: Perfect consistency (all time series are identical)
                ~0.5: Moderate consistency
                0.0: No consistency (single point or completely inconsistent)

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get all time series in the cluster
        y_vals = self.td.get_timeseries(
            self.var,
            cluster_id=cluster_id,
        )

        # Convert to array
        time_series = np.array(y_vals)

        if len(time_series) <= 1:
            return 0.0  # Not enough data to assess consistency

        # Compute R² similarity matrix # prevent warning about division by zero variance
        with np.errstate(divide="ignore", invalid="ignore"):
            r_matrix = np.corrcoef(time_series)
        r_squared_matrix = np.nan_to_num(r_matrix**2)

        # Compute distance matrix
        distance_matrix = 1 - r_squared_matrix

        # Ensure symmetry and 0 diagonal
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        # Linkage for hierarchical clustering
        Z = linkage(squareform(distance_matrix), method="ward")
        d = len(time_series) - 1
        R = inconsistent(Z, d=d)

        # Inconsistency at top level
        inconsistency_value = R[-1, -1]

        # Convert to consistency score
        if inconsistency_value < 1e-10:  # to avoid numerical issues
            return 1.0
        else:
            return 1.0 / inconsistency_value

    def score_spatial_autocorrelation(
        self,
        cluster_id: int,
    ) -> float:
        """Computes average pairwise similarity (R²) between all time series in a cluster.

        This measures how spatially coherent the cluster behavior is.

        The score is calculated by:
        1. Getting all time series for cells in the cluster
        2. Computing pairwise R² correlations between all time series
        3. Taking the mean of the upper triangle of the correlation matrix

        Args:
            cluster_id: ID of the cluster to evaluate.

        Returns:
            float: Similarity score between 0-1, where:
                1.0: Perfect similarity (all time series identical)
                ~0.5: Moderate spatial coherence
                0.0: No similarity (completely uncorrelated)

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get all time series in the cluster
        y_vals = self.td.get_timeseries(
            self.var,
            cluster_id=cluster_id,
        )

        # Convert to array
        time_series = np.array(y_vals)

        if len(time_series) <= 1:
            return 0.0  # Not enough data to assess similarity

        # Compute R² similarity matrix # prevent warning about division by zero variance
        with np.errstate(divide="ignore", invalid="ignore"):
            r_matrix = np.corrcoef(time_series)
        r_squared_matrix = np.nan_to_num(r_matrix**2)

        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices_from(r_squared_matrix, k=1)
        avg_similarity = float(np.mean(r_squared_matrix[upper_triangle_indices]))

        return avg_similarity

    def score_nonlinearity(
        self,
        cluster_id: int,
        aggregation: Literal["mean", "sum", "std", "median", "percentile"] = "mean",
        percentile: Optional[float] = None,
        normalise_against_unclustered: bool = False,
    ) -> float:
        """Computes nonlinearity of a cluster's aggregated time series using RMSE from a linear fit.

        The score measures how much the time series deviates from a linear trend.

        When normalise_against_unclustered=True:
            - Score > 1: Cluster is more nonlinear than typical unclustered behavior
            - Score ≈ 1: Cluster has similar nonlinearity to unclustered data
            - Score < 1: Cluster is more linear than unclustered data
        When normalise_against_unclustered=False:
            - Returns raw RMSE (0 = perfectly linear, higher = more nonlinear)
            - Useful for comparing clusters to each other

        Args:
            cluster_id: Cluster ID to evaluate.
            aggregation: How to aggregate spatial data:
                - "mean": Average across space
                - "median": Median across space
                - "sum": Sum across space
                - "std": Standard deviation across space
                - "percentile": Percentile across space (requires percentile arg)
            percentile: Percentile value between 0–1 (only used if aggregation="percentile")
            normalise_against_unclustered: If True, normalise score by average RMSE of unclustered points.
                This helps identify clusters that stand out from background behavior.

        Returns:
            float: Nonlinearity score. Higher means more nonlinear behavior.
                Interpretation depends on normalise_against_unclustered parameter.

        References:
            Kobe De Maeyer Master Thesis (2025)
        """
        # Get aggregated cluster time series
        yvals = self.td.get_timeseries(
            self.var,
            cluster_id=cluster_id,
            aggregation=aggregation,
            percentile=percentile,
            normalize="max",
        ).values

        xvals = self.td.numeric_time_values  # use numeric time values for fitting
        if len(xvals) != len(yvals):
            raise ValueError("Time and data dimensions do not match.")

        # Fit linear model with polyfit
        coeffs = np.polyfit(xvals, yvals, 1)
        predicted = np.polyval(coeffs, xvals)

        # RMSE for clustered series
        rmse_cluster = float(np.sqrt(np.mean((yvals - predicted) ** 2)))

        if not normalise_against_unclustered:
            return rmse_cluster

        # Get unclustered time series (raw, no aggregation)
        y_unclustered = self.td.get_timeseries(
            self.var,
            cluster_id=-1,
            aggregation="raw",
        ).values

        unclustered_rmses = []
        for ts in y_unclustered:
            ts = ts.astype(np.float32)
            if np.all(np.isnan(ts)) or np.nanmax(ts) - np.nanmin(ts) == 0:
                continue
            norm_ts = (ts - np.nanmin(ts)) / (
                np.nanmax(ts) - np.nanmin(ts)
            )  # Normalise each trajectory individually
            valid = np.isfinite(norm_ts)
            if valid.sum() < 2:
                continue
            coeffs = np.polyfit(xvals[valid], norm_ts[valid], 1)
            pred = np.polyval(coeffs, xvals)
            rmse = np.sqrt(np.mean((norm_ts[valid] - pred[valid]) ** 2))
            unclustered_rmses.append(rmse)

        if len(unclustered_rmses) == 0:
            return rmse_cluster  # No valid unclustered points

        avg_rmse_unclustered = np.nanmean(unclustered_rmses)
        if avg_rmse_unclustered < 1e-10:
            avg_rmse_unclustered = 1e-10  # avoid divide-by-zero

        # Return normalized nonlinearity
        return float(rmse_cluster / avg_rmse_unclustered)

    def score_overview(
        self,
        exclude_noise: bool = True,
        shift_threshold: float = 0.0,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute all available scores for every cluster and return as a pandas DataFrame.

        This function computes all scoring methods defined in score_dictionary for each cluster
        and returns the results in a structured DataFrame format, similar to the consensus summary.
        Includes cluster size, spatial means, shift time statistics, and an aggregate score.

        Args:
            exclude_noise: Whether to exclude noise points (cluster ID -1). Defaults to True.
            shift_threshold: Minimum shift threshold for computing transition times. Defaults to 0.0.
            **kwargs: Additional keyword arguments passed to scoring methods. These will be
                applied to all scoring methods that accept them. Common parameters include:
                - aggregation: Aggregation method for methods that support it (default: "mean")
                - percentile: Percentile value for percentile aggregation
                - normalize: Normalization method for score_heaviside
                - normalise_against_unclustered: Boolean for score_nonlinearity (default: False)

        Returns:
            pd.DataFrame: DataFrame with one row per cluster containing:
                - cluster_id: Cluster identifier
                - All score columns from score_dictionary
                - size: Number of space-time grid cells in the cluster
                - mean_{spatial_dim0}: Average spatial coordinate for first dimension
                - mean_{spatial_dim1}: Average spatial coordinate for second dimension
                - mean_shift_time: Mean transition time for the cluster
                - std_shift_time: Standard deviation of transition times within the cluster
                - aggregate_score: Product of all score values

        Example:
            >>> td.stats(var="temperature").score_overview()
        """
        # Get all cluster IDs
        cluster_ids = self.td.get_cluster_ids(self.var, exclude_noise=exclude_noise)

        if len(cluster_ids) == 0:
            # Return empty DataFrame with expected columns
            spatial_dims = self.td.space_dims
            cols = (
                ["cluster_id"]
                + list(score_dictionary.keys())
                + ["size", f"mean_{spatial_dims[0]}", f"mean_{spatial_dims[1]}"]
                + ["mean_shift_time", "std_shift_time", "aggregate_score"]
            )
            return pd.DataFrame({c: [] for c in cols})

        # Get spatial dimensions
        spatial_dims = self.td.space_dims
        sd0, sd1 = spatial_dims[0], spatial_dims[1]

        # Initialize results dictionary
        results = {"cluster_id": cluster_ids.tolist()}

        # Compute each score for all clusters
        score_columns = []
        for score_name, method_name in score_dictionary.items():
            method = getattr(self, method_name)
            scores = []

            for cluster_id in cluster_ids:
                try:
                    # Call method with cluster_id and any additional kwargs
                    score = method(cluster_id, **kwargs)
                    # Handle tuple returns (e.g., score_heaviside with return_score_fit=True)
                    if isinstance(score, tuple):
                        score = score[0]  # Take the first element (the score)
                    scores.append(float(score))
                except Exception as e:
                    logger.warning(
                        "score_overview failed for score '%s' (method '%s') on cluster_id=%s: %s",
                        score_name,
                        method_name,
                        cluster_id,
                        e,
                    )
                    # If computation fails, store NaN
                    scores.append(np.nan)

            results[score_name] = scores
            score_columns.append(score_name)

        # Compute cluster size and spatial means
        sizes = []
        mean_sd0 = []
        mean_sd1 = []

        for cluster_id in cluster_ids:
            # Get cluster mask (3D: time x space x space)
            mask = self.td.get_cluster_mask(self.var, cluster_id)
            # Size is total number of space-time cells
            size = int(mask.sum().values)
            sizes.append(size)

            # Compute spatial means using 2D spatial mask (any time)
            spatial_mask = mask.any(dim=self.td.time_dim)
            if spatial_mask.sum() > 0:
                # Get spatial coordinates
                coords_sd0 = self.td.data[sd0].where(spatial_mask)
                coords_sd1 = self.td.data[sd1].where(spatial_mask)
                mean_sd0.append(float(coords_sd0.mean(skipna=True).values))
                mean_sd1.append(float(coords_sd1.mean(skipna=True).values))
            else:
                mean_sd0.append(np.nan)
                mean_sd1.append(np.nan)

        results["size"] = sizes
        results[f"mean_{sd0}"] = mean_sd0
        results[f"mean_{sd1}"] = mean_sd1

        # Compute shift time statistics
        # Get shift variable for this cluster variable
        try:
            shift_var = self.td.data[self.var].attrs.get("shifts_variable")
            if shift_var is None:
                # Try to get shifts variable using the method
                shift_var = self.td.get_shifts(self.var).name
        except Exception:
            shift_var = None

        mean_shift_times = []
        std_shift_times = []

        if shift_var is not None:
            try:
                # Compute transition time map (2D spatial array)
                transition_time_map = self.td.stats(
                    shift_var
                ).time.compute_transition_time(shift_threshold=shift_threshold)

                for cluster_id in cluster_ids:
                    # Get spatial mask for this cluster (2D boolean mask)
                    cluster_mask_2d = self.td.get_cluster_mask_spatial(
                        self.var, cluster_id
                    )

                    # Extract transition times for this cluster
                    cluster_transition_times = transition_time_map.where(
                        cluster_mask_2d
                    ).values

                    # Filter out NaN values
                    valid_times = cluster_transition_times[
                        np.isfinite(cluster_transition_times)
                    ]

                    if len(valid_times) > 0:
                        mean_shift_times.append(float(np.mean(valid_times)))
                        std_shift_times.append(float(np.std(valid_times)))
                    else:
                        mean_shift_times.append(np.nan)
                        std_shift_times.append(np.nan)
            except Exception:
                # If computation fails, fill with NaN
                mean_shift_times = [np.nan] * len(cluster_ids)
                std_shift_times = [np.nan] * len(cluster_ids)
        else:
            # No shift variable available
            mean_shift_times = [np.nan] * len(cluster_ids)
            std_shift_times = [np.nan] * len(cluster_ids)

        results["mean_shift_time"] = mean_shift_times
        results["std_shift_time"] = std_shift_times

        # Create DataFrame
        df = pd.DataFrame(results)

        # Compute aggregate_score as product of all scores
        aggregate_scores = []
        for idx in range(len(df)):
            score_values = [df.loc[idx, col] for col in score_columns]
            # Filter out NaN values before computing product
            valid_scores = [s for s in score_values if not np.isnan(s)]
            if len(valid_scores) > 0:
                aggregate_scores.append(float(np.prod(valid_scores)))
            else:
                aggregate_scores.append(np.nan)

        df["aggregate_score"] = aggregate_scores

        return df

    def aggregate_cluster_scores(
        self,
        cluster_ids: list[int],
        score_method: str,
        aggregation: Union[str, Callable] = "mean",
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """Compute a score for multiple clusters and aggregate the results.

        Args:
            cluster_ids: List of cluster IDs
            score_method: Name of the scoring method (e.g., "score_nonlinearity")
            aggregation: "mean", "median", "weighted", or custom function
            weights: Weights for each cluster (if aggregation="weighted")
            **kwargs: Arguments passed to the scoring method

        Returns:
            float: Aggregated score across all clusters
        """
        method = getattr(self, score_method)
        scores = [method(cid, **kwargs) for cid in cluster_ids]

        if callable(aggregation):
            return aggregation(scores)
        elif aggregation == "mean":
            return float(np.mean(scores))
        elif aggregation == "median":
            return float(np.median(scores))
        elif aggregation == "weighted":
            return float(np.average(scores, weights=weights))
        else:
            return 0

    def size_table(
        self,
        cluster_ids: list[int] | range | None = None,
        *,
        events=None,
        exclude_noise: bool = True,
    ) -> pd.DataFrame:
        """Batch spacetime cell counts per cluster (same as ``mask.sum()``).

        Args:
            cluster_ids: Cluster ids to include. Defaults to all non-noise ids.
            events: Optional precomputed :class:`ClusterLabelEvents`.
            exclude_noise: Used only when ``cluster_ids`` is None.
        """
        from toad.postprocessing.stats.cluster_events import (
            _group_slices,
            load_cluster_label_events,
        )

        if cluster_ids is None:
            cluster_ids = [
                int(c)
                for c in self.td.get_cluster_ids(self.var, exclude_noise=exclude_noise)
            ]
        else:
            cluster_ids = [int(c) for c in cluster_ids]

        if events is None:
            events = load_cluster_label_events(self.td, self.var, cluster_ids)

        counts: dict[int, int] = {int(cid): 0 for cid in cluster_ids}
        if events.labels.size:
            order = np.argsort(events.labels, kind="mergesort")
            lab_s = events.labels[order]
            for cid, s, e in _group_slices(lab_s):
                if cid in counts:
                    counts[cid] = int(e - s)

        return pd.DataFrame(
            [{"cluster_id": int(cid), "size": counts[int(cid)]} for cid in cluster_ids]
        )
