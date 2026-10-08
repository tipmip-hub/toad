import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN


class SpaceTimeDBSCAN(ClusterMixin, BaseEstimator):
    """DBSCAN with separate eps for spatial and temporal dimensions.

    Performs its own scaling of coordinates; TOAD's time scaling is skipped when
    this clusterer is used (via the skip_time_scaling protocol).

    Parameters
    ----------
    spatial_eps : float
        Eps for spatial dimensions (same units as spatial coordinates passed by TOAD).
    temporal_eps : float
        Eps for the temporal dimension (same units as time coordinates passed by TOAD).
    min_samples : int
        Minimum samples per cluster (DBSCAN min_samples).
    """

    skip_time_scaling = True  # Protocol: TOAD skips its time scaling when True

    def __init__(
        self,
        spatial_eps: float,
        temporal_eps: float,
        min_samples: int = 5,
    ):
        if spatial_eps is None or temporal_eps is None:
            raise ValueError("spatial_eps and temporal_eps are required")
        if spatial_eps <= 0 or temporal_eps <= 0:
            raise ValueError("spatial_eps and temporal_eps must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.spatial_eps = spatial_eps
        self.temporal_eps = temporal_eps
        self.min_samples = min_samples

    def fit_predict(self, X: np.ndarray, y=None, **kwargs):
        # TODO(clustering-weights): ``y`` (shift magnitudes from TOAD) is required here but
        # not used. See TODO in toad.clustering.compute_clusters.
        assert y is not None, "y must be provided"

        X_scaled = X.copy()
        X_scaled[:, 1:] = X[:, 1:] / self.spatial_eps
        X_scaled[:, 0] = X[:, 0] / self.temporal_eps

        dbscan = DBSCAN(eps=1.0, min_samples=self.min_samples, metric="euclidean")
        return dbscan.fit_predict(X_scaled)
