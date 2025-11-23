import numpy as np
from sklearn.decomposition import IncrementalPCA
from collections import deque

class AdvancedBucketDetector:
    """
    Enhanced bucket detector for identifying poisoning attacks in federated learning.
    Combines multiple statistical checks to detect anomalous model updates,
    with particular focus on targeted label flipping attacks.
    """

    def __init__(
        self,
        n_features: int,
        sample_size: int = None,
        memory_size: int = 100,
        sensitivity: float = 1.5,     # Increased sensitivity for better detection
        vote_threshold: float = 0.2,  # Lower threshold to catch subtler attacks
        pca_components: int = 5,
        warmup_rounds: int = 2        # Minimal warmup period
    ):
        # Core parameters
        self.n_features = sample_size if sample_size is not None else n_features
        self.memory_size = memory_size
        self.sensitivity = sensitivity
        self.vote_threshold = vote_threshold
        self.warmup_rounds = warmup_rounds
        self.current_round = 0
        
        # Robust statistics tracking
        self.median = np.zeros(self.n_features, dtype=float)
        self.mad = np.ones(self.n_features, dtype=float)
        self.count = 0
        
        # PCA for spectral analysis
        self.pca = IncrementalPCA(n_components=min(pca_components, self.n_features))
        self.history = deque(maxlen=memory_size)
        
        # Previous global model reference
        self.last_global = None
        
        # Metric history trackers
        self.metric_hist = {
            'mad95': deque(maxlen=memory_size),
            'phi_t': deque(maxlen=memory_size),
            'phi_s': deque(maxlen=memory_size),
            'eps': deque(maxlen=memory_size),
            'skew': deque(maxlen=memory_size),
            'kurt': deque(maxlen=memory_size),
            'magnitude': deque(maxlen=memory_size),
        }
        
        # Previous metrics for shift detection
        self.prev_metrics = {
            'mad95': 0, 'phi_t': 1.0, 'phi_s': 1.0, 'eps': 0,
            'skew': 0, 'kurt': 0, 'magnitude': 0
        }
        
        # Consecutive anomaly tracking
        self.consecutive_anomalies = 0
        
        # Enable detailed debugging
        self.debug = True

    def _update_robust_stats(self, x):
        """Update robust statistics with new sample"""
        self.count += 1
        delta = x - self.median
        self.median += delta / self.count
        self.mad += (np.abs(x - self.median) - self.mad) / self.count

    def _dynamic_threshold(self, key):
        """Calculate adaptive thresholds based on metric history"""
        hist = np.array(self.metric_hist[key])
        
        # Enable checks once we have minimal data
        if len(hist) < 3:
            return (-np.inf if key.startswith('phi') else np.inf)
            
        # Calculate mean and std, with protection against zero std
        μ = hist.mean()
        σ = max(hist.std(), 1e-6)
        
        # For similarity metrics, lower values are suspicious
        # For error metrics, higher values are suspicious
        if key.startswith('phi'):
            return μ - self.sensitivity * σ
        else:
            return μ + self.sensitivity * σ

    def analyze(self, x, other_clean=None):
        """
        Analyze a model update for signs of poisoning.
        
        Args:
            x: The model parameters to analyze
            other_clean: List of known clean model parameters for comparison
            
        Returns:
            Tuple of (suspicion_score, flags_dict, metrics_dict)
        """
        self.current_round += 1
        
        # -----------------------------------
        # 1. Compute basic statistical metrics
        # -----------------------------------
        
        # MAD z-scores for outlier detection
        z = np.abs(x - self.median) / (1.4826 * self.mad + 1e-9)
        mad95 = float(np.percentile(z, 95))
        
        # Temporal similarity (cosine with previous global model)
        phi_t = 1.0
        if self.last_global is not None:
            phi_t = float((x @ self.last_global) / 
                         (np.linalg.norm(x) * np.linalg.norm(self.last_global) + 1e-9))
        
        # Spatial similarity (cosine with other clients)
        phi_s = 1.0
        if other_clean and len(other_clean) > 0:
            sims = [(x @ u) / (np.linalg.norm(x) * np.linalg.norm(u) + 1e-9) 
                   for u in other_clean]
            phi_s = float(np.mean(sims))
        
        # Spectral error (reconstruction error using PCA)
        eps = 0.0
        if hasattr(self.pca, 'components_') and len(self.history) >= self.pca.n_components:
            try:
                proj = self.pca.transform(x.reshape(1, -1))
                recon = self.pca.inverse_transform(proj)
                eps = float(np.linalg.norm(x - recon.ravel()))
            except:
                eps = 0.0
        
        # Update magnitude
        magnitude = float(np.linalg.norm(x))
        
        # Distribution shape metrics
        mean_x = np.mean(x)
        std_x = max(np.std(x), 1e-9)  # Avoid division by zero
        
        # Skewness (3rd moment)
        skew = float(np.mean(((x - mean_x) / std_x) ** 3))
        
        # Kurtosis (4th moment, excess kurtosis)
        kurt = float(np.mean(((x - mean_x) / std_x) ** 4) - 3)
        
        # -----------------------------------
        # 2. Calculate metric shifts
        # -----------------------------------
        shifts = {}
        for k, v in [('mad95', mad95), ('phi_t', phi_t), ('phi_s', phi_s), 
                    ('eps', eps), ('skew', skew), ('kurt', kurt), 
                    ('magnitude', magnitude)]:
            # Calculate shift from previous value
            prev = self.prev_metrics[k]
            
            # Absolute change for similarity metrics
            if k.startswith('phi'):
                shift = abs(v - prev)
            # Relative change for others
            else:
                # Avoid division by zero
                shift = abs(v - prev) / (abs(prev) + 1e-9)
                
            shifts[k] = shift
            self.prev_metrics[k] = v
        
        # -----------------------------------
        # 3. Store metrics history
        # -----------------------------------
        for k, v in [('mad95', mad95), ('phi_t', phi_t), ('phi_s', phi_s), 
                    ('eps', eps), ('skew', abs(skew)), ('kurt', abs(kurt)), 
                    ('magnitude', magnitude)]:
            self.metric_hist[k].append(v)
        
        # -----------------------------------
        # 4. Calculate detection thresholds
        # -----------------------------------
        τ_m = self._dynamic_threshold('mad95')
        τ_t = self._dynamic_threshold('phi_t')
        τ_s = self._dynamic_threshold('phi_s')
        τ_e = self._dynamic_threshold('eps')
        τ_skew = self._dynamic_threshold('skew')
        τ_kurt = self._dynamic_threshold('kurt')
        τ_mag = self._dynamic_threshold('magnitude')
        
        # -----------------------------------
        # 5. Compute anomaly flags
        # -----------------------------------
        
        # Standard statistic flags
        std_flags = {
            'mad': mad95 > τ_m,
            'temp': phi_t < τ_t,
            'spat': phi_s < τ_s,
            'spec': eps > τ_e,
            'skew': abs(skew) > τ_skew,
            'kurt': abs(kurt) > τ_kurt,
            'magnitude': magnitude > τ_mag,
        }
        
        # Shift-based flags (detect sudden changes)
        shift_threshold = 0.25  # Significant change threshold
        shift_flags = {
            'temp_shift': shifts['phi_t'] > shift_threshold,
            'spat_shift': shifts['phi_s'] > shift_threshold,
            'mag_shift': shifts['magnitude'] > 0.5,  # 50% change in magnitude
        }
        
        # Combine all flags
        flags = {**std_flags, **shift_flags}
        
        # -----------------------------------
        # 6. Compute suspicion score & verdict
        # -----------------------------------
        
        # Calculate suspicion score (proportion of triggered flags)
        # Only count valid flags (not None)
        valid_flags = {k: v for k, v in flags.items() if v is not False and v is not None}
        S = sum(valid_flags.values()) / len(flags) if flags else 0
        
        # Boost suspicion score for consecutive anomalies
        if S > self.vote_threshold * 0.8:  # Close to threshold
            self.consecutive_anomalies += 1
            # Apply boosting for repeated near-threshold anomalies
            if self.consecutive_anomalies > 1:
                S = min(1.0, S * (1.0 + 0.1 * self.consecutive_anomalies))
        else:
            self.consecutive_anomalies = 0
        
        # -----------------------------------
        # 7. Output debug information
        # -----------------------------------
        if self.debug:
            print(f"\n--- BUCKET ANALYSIS: Round {self.current_round} ---")
            print(f"Suspicion Score: {S:.3f} (Threshold: {self.vote_threshold:.3f})")
            print(f"Basic Metrics:")
            print(f"  MAD95: {mad95:.3f} (τ={τ_m:.3f}) → {'ANOMALY' if std_flags['mad'] else 'normal'}")
            print(f"  Temporal Similarity: {phi_t:.3f} (τ={τ_t:.3f}) → {'ANOMALY' if std_flags['temp'] else 'normal'}")
            print(f"  Spatial Similarity: {phi_s:.3f} (τ={τ_s:.3f}) → {'ANOMALY' if std_flags['spat'] else 'normal'}")
            print(f"  Magnitude: {magnitude:.3f} (τ={τ_mag:.3f}) → {'ANOMALY' if std_flags['magnitude'] else 'normal'}")
            print(f"Shape Metrics:")
            print(f"  Skewness: {skew:.3f} (τ={τ_skew:.3f}) → {'ANOMALY' if std_flags['skew'] else 'normal'}")
            print(f"  Kurtosis: {kurt:.3f} (τ={τ_kurt:.3f}) → {'ANOMALY' if std_flags['kurt'] else 'normal'}")
            print(f"Shift Metrics:")
            print(f"  Temporal Shift: {shifts['phi_t']:.3f} → {'ANOMALY' if shift_flags['temp_shift'] else 'normal'}")
            print(f"  Spatial Shift: {shifts['phi_s']:.3f} → {'ANOMALY' if shift_flags['spat_shift'] else 'normal'}")
            print(f"  Magnitude Shift: {shifts['magnitude']:.3f} → {'ANOMALY' if shift_flags['mag_shift'] else 'normal'}")
            print(f"VERDICT: {'POISONED' if S > self.vote_threshold else 'CLEAN'}")
        
        # -----------------------------------
        # 8. Update statistics for clean samples
        # -----------------------------------
        if S <= self.vote_threshold:
            self._update_robust_stats(x)
            self.history.append(x)
            # Only update PCA with clean samples
            if len(self.history) >= self.pca.n_components:
                try:
                    self.pca.partial_fit(x.reshape(1, -1))
                except:
                    pass
            
            # Update global reference for next round
            self.last_global = x.copy()
        
        # -----------------------------------
        # 9. Return results
        # -----------------------------------
        metrics = {
            'mad95': mad95,
            'phi_t': phi_t,
            'phi_s': phi_s,
            'eps': eps,
            'skew': skew,
            'kurt': kurt,
            'magnitude': magnitude,
            'temp_shift': shifts['phi_t'],
            'spat_shift': shifts['phi_s'],
            'mag_shift': shifts['magnitude'],
            'consecutive_anomalies': self.consecutive_anomalies
        }
        
        return S, flags, metrics