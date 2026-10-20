"""Published GxE benchmark models run under the same information protocol.

The implementations here deliberately use only SNPs, daily environmental records
available up to the requested prediction day, and training-set yield labels.  They
are data-matched reimplementations, not claims of bitwise reproduction of external
software packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

from .data import GPACEData
from .features import ProjectedInputs, interaction_design
from .models import YieldHeadRegressor


def _pairwise_sqeuclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left**2, axis=1, keepdims=True)
    right_norm = np.sum(right**2, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


def _rbf(left: np.ndarray, right: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-_pairwise_sqeuclidean(left, right) / max(scale, 1e-8))


def _median_scale(values: np.ndarray) -> float:
    distance = _pairwise_sqeuclidean(values, values)
    positive = distance[distance > 1e-10]
    return float(np.median(positive)) if len(positive) else 1.0


def _yield(data: GPACEData, indices: np.ndarray) -> np.ndarray:
    return data.records.loc[indices, "grain.yield"].to_numpy(dtype=np.float64)


@dataclass
class ECGBLUP:
    """Environmental-covariate GBLUP with a linear genomic-by-environment term.

    This is a scalable RR-BLUP formulation. Genomic and daily-environment
    covariance are represented by training-fitted principal components and the
    corresponding GxE basis is penalised separately. It provides a transparent
    classical comparator to nonlinear reaction-norm and neural-network models.
    """

    genotype_components: int = 64
    environment_components: int = 12
    genotype_alpha: float = 10.0
    environment_alpha: float = 1.0
    interaction_alpha: float = 30.0
    random_state: int = 2026

    def fit(self, data: GPACEData, indices: np.ndarray) -> "ECGBLUP":
        self.inputs_ = ProjectedInputs(
            self.genotype_components, self.environment_components, self.random_state
        ).fit(data, indices)
        genotype, environment = self.inputs_.rows(data, indices)
        self.genotype_scaler_ = StandardScaler().fit(genotype)
        self.environment_scaler_ = StandardScaler().fit(environment)
        design = self._design(
            self.genotype_scaler_.transform(genotype),
            self.environment_scaler_.transform(environment),
        )
        target = _yield(data, indices)
        self.target_mean_ = float(target.mean())
        penalties = np.concatenate(
            [
                np.full(self.genotype_features_, self.genotype_alpha),
                np.full(self.environment_features_, self.environment_alpha),
                np.full(self.interaction_features_, self.interaction_alpha),
            ]
        )
        # Separate variance penalties are the marker, environment and GxE
        # random-effect analogue of a multi-environment GBLUP model.
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += penalties
        self.coefficients_ = np.linalg.solve(gram, design.T @ (target - self.target_mean_))
        return self

    def _design(self, genotype: np.ndarray, environment: np.ndarray) -> np.ndarray:
        interaction = np.einsum("bi,bj->bij", genotype, environment).reshape(len(genotype), -1)
        self.genotype_features_ = genotype.shape[1]
        self.environment_features_ = environment.shape[1]
        self.interaction_features_ = interaction.shape[1]
        return np.concatenate([genotype, environment, interaction], axis=1).astype(np.float64)

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        genotype, environment = self.inputs_.rows(data, indices)
        design = self._design(
            self.genotype_scaler_.transform(genotype),
            self.environment_scaler_.transform(environment),
        )
        return self.target_mean_ + design @ self.coefficients_


@dataclass
class KernelReactionNorm:
    """Environmental-relatedness reaction norm (Jarquín-style RKHS baseline)."""

    genotype_components: int = 32
    environment_components: int = 12
    alpha: float = 1.0
    genotype_weight: float = 1.0
    environment_weight: float = 1.0
    interaction_weight: float = 1.0
    random_state: int = 2026

    def fit(self, data: GPACEData, indices: np.ndarray) -> "KernelReactionNorm":
        self.inputs_ = ProjectedInputs(
            self.genotype_components, self.environment_components, self.random_state
        ).fit(data, indices)
        genotype, environment = self.inputs_.rows(data, indices)
        self.genotype_scaler_ = StandardScaler().fit(genotype)
        self.environment_scaler_ = StandardScaler().fit(environment)
        self.train_genotype_ = self.genotype_scaler_.transform(genotype)
        self.train_environment_ = self.environment_scaler_.transform(environment)
        self.g_scale_ = _median_scale(self.train_genotype_)
        self.e_scale_ = _median_scale(self.train_environment_)
        kernel = self._kernel(self.train_genotype_, self.train_environment_)
        self.target_mean_ = float(_yield(data, indices).mean())
        self.model_ = KernelRidge(alpha=self.alpha, kernel="precomputed").fit(
            kernel, _yield(data, indices) - self.target_mean_
        )
        return self

    def _kernel(self, genotype: np.ndarray, environment: np.ndarray) -> np.ndarray:
        genetic = _rbf(genotype, self.train_genotype_, self.g_scale_)
        environmental = _rbf(environment, self.train_environment_, self.e_scale_)
        return (
            self.genotype_weight * genetic
            + self.environment_weight * environmental
            + self.interaction_weight * genetic * environmental
        )

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        genotype, environment = self.inputs_.rows(data, indices)
        genotype = self.genotype_scaler_.transform(genotype)
        environment = self.environment_scaler_.transform(environment)
        return self.target_mean_ + self.model_.predict(self._kernel(genotype, environment))


@dataclass
class RFFReactionNorm:
    """Scalable random-Fourier approximation of the RKHS reaction norm.

    ``KernelReactionNorm`` is the exact Jarquín-style kernel implementation,
    which is appropriate for the NG data but requires an O(n^2) Gram matrix.
    G2F has roughly 78,000 training GxE means, so this approximation preserves
    the same genomic, environmental and GxE-kernel structure while fitting a
    linear ridge model in an explicit random-Fourier feature space.
    """

    genotype_components: int = 32
    environment_components: int = 12
    rff_components: int = 128
    alpha: float = 10.0
    genotype_weight: float = 1.0
    environment_weight: float = 1.0
    interaction_weight: float = 1.0
    random_state: int = 2026

    def fit(self, data: GPACEData, indices: np.ndarray) -> "RFFReactionNorm":
        self.inputs_ = ProjectedInputs(
            self.genotype_components, self.environment_components, self.random_state
        ).fit(data, indices)
        genotype, environment = self.inputs_.rows(data, indices)
        self.genotype_scaler_ = StandardScaler().fit(genotype)
        self.environment_scaler_ = StandardScaler().fit(environment)
        genotype = self.genotype_scaler_.transform(genotype)
        environment = self.environment_scaler_.transform(environment)
        self.g_scale_ = _median_scale(genotype)
        self.e_scale_ = _median_scale(environment)
        self.genotype_rff_ = RBFSampler(
            gamma=1.0 / max(self.g_scale_, 1e-8),
            n_components=self.rff_components,
            random_state=self.random_state,
        ).fit(genotype)
        self.environment_rff_ = RBFSampler(
            gamma=1.0 / max(self.e_scale_, 1e-8),
            n_components=self.rff_components,
            random_state=self.random_state + 1,
        ).fit(environment)
        design = self._design(genotype, environment)
        self.target_mean_ = float(_yield(data, indices).mean())
        self.model_ = Ridge(alpha=self.alpha).fit(
            design, _yield(data, indices) - self.target_mean_
        )
        return self

    def _design(self, genotype: np.ndarray, environment: np.ndarray) -> np.ndarray:
        genotype_features = self.genotype_rff_.transform(genotype).astype(np.float32)
        environment_features = self.environment_rff_.transform(environment).astype(np.float32)
        interaction_features = genotype_features * environment_features
        return np.concatenate(
            [
                self.genotype_weight * genotype_features,
                self.environment_weight * environment_features,
                self.interaction_weight * interaction_features,
            ],
            axis=1,
        )

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        genotype, environment = self.inputs_.rows(data, indices)
        genotype = self.genotype_scaler_.transform(genotype)
        environment = self.environment_scaler_.transform(environment)
        return self.target_mean_ + self.model_.predict(self._design(genotype, environment))


@dataclass
class CERISJGRA:
    """CERIS-selected environmental index plus joint genomic reaction norm.

    The environmental index is selected exclusively from training environments by
    scanning daily-variable windows against their training-set mean yield.  Genomic
    main effects and genomic slopes are then fitted jointly with ridge regression.
    """

    genotype_components: int = 32
    alpha: float = 10.0
    min_window: int = 7
    max_window: int = 35
    window_step: int = 7
    random_state: int = 2026

    def fit(self, data: GPACEData, indices: np.ndarray) -> "CERISJGRA":
        self.inputs_ = ProjectedInputs(
            self.genotype_components, 2, self.random_state
        ).fit(data, indices)
        self._select_index(data, indices)
        genotype, _ = self.inputs_.rows(data, indices)
        self.geno_scaler_ = StandardScaler().fit(genotype)
        genotype = self.geno_scaler_.transform(genotype)
        environmental_index = self._environment_index(data, indices)
        self.index_mean_ = float(environmental_index.mean())
        self.index_scale_ = float(environmental_index.std()) or 1.0
        environmental_index = (environmental_index - self.index_mean_) / self.index_scale_
        design = np.concatenate(
            [genotype, environmental_index[:, None], genotype * environmental_index[:, None]],
            axis=1,
        )
        self.model_ = Ridge(alpha=self.alpha).fit(design, _yield(data, indices))
        return self

    def _select_index(self, data: GPACEData, indices: np.ndarray) -> None:
        rows = data.records.loc[indices]
        grouped = rows.groupby("environment_index", sort=False)["grain.yield"].mean()
        environment_indices = grouped.index.to_numpy(dtype=int)
        target = grouped.to_numpy(dtype=np.float64)
        sequences = data.environment_sequences[environment_indices]
        lengths = data.environment_sequence_lengths[environment_indices]
        best: tuple[float, int, int, int, float] | None = None
        # A CERIS candidate must exist in every fitting environment.  Restricting
        # the scan to the shortest training sequence avoids implicit imputation of
        # late-season windows for environments with shorter records.
        max_day = int(min(lengths.min(), sequences.shape[1]))
        for variable in range(sequences.shape[-1]):
            for width in range(self.min_window, min(self.max_window, max_day) + 1, self.window_step):
                for start in range(0, max_day - width + 1, self.window_step):
                    values = []
                    for pos in range(len(lengths)):
                        segment = sequences[pos, start : start + width, variable]
                        values.append(float(np.nanmean(segment)) if np.isfinite(segment).any() else np.nan)
                    values = np.asarray(values, dtype=np.float64)
                    if not np.isfinite(values).all() or np.std(values) < 1e-9:
                        continue
                    score = abs(float(np.corrcoef(values, target)[0, 1]))
                    if np.isfinite(score) and (best is None or score > best[0]):
                        best = (score, variable, start, width, float(np.corrcoef(values, target)[0, 1]))
        if best is None:
            raise ValueError("CERIS could not identify an environmental index")
        self.index_correlation_, self.variable_, self.start_day_, self.window_, self.index_direction_ = best

    def _environment_index(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        environment_indices = data.records.loc[indices, "environment_index"].to_numpy(dtype=int)
        values = []
        for environment_index in environment_indices:
            length = int(data.environment_sequence_lengths[environment_index])
            stop = min(self.start_day_ + self.window_, length)
            values.append(
                np.nanmean(
                    data.environment_sequences[environment_index, self.start_day_ : stop, self.variable_]
                )
            )
        return np.asarray(values, dtype=np.float64)

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        genotype, _ = self.inputs_.rows(data, indices)
        genotype = self.geno_scaler_.transform(genotype)
        environmental_index = self._environment_index(data, indices)
        environmental_index = (environmental_index - self.index_mean_) / self.index_scale_
        design = np.concatenate(
            [genotype, environmental_index[:, None], genotype * environmental_index[:, None]],
            axis=1,
        )
        return self.model_.predict(design)


@dataclass
class AutoGSStyle:
    """Training-only marker-screening and environment-aware AutoML baseline.

    This follows the Auto-GS design principles (marker selection, reduced
    environmental parameters, automated yield-head selection), while remaining an
    independent implementation adapted to this dataset rather than the authors'
    original software workflow.
    """

    n_markers: int = 512
    environment_components: int = 12
    head: str = "catboost"
    head_params: dict[str, Any] | None = None
    random_state: int = 2026
    device: str = "auto"

    def fit(self, data: GPACEData, indices: np.ndarray) -> "AutoGSStyle":
        rows = data.records.loc[indices]
        residual = rows["grain.yield"] - rows.groupby("Experiment")["grain.yield"].transform("mean")
        genotype_target = residual.groupby(rows["genotype_index"]).mean()
        genotype_indices = genotype_target.index.to_numpy(dtype=int)
        matrix = data.genotype_matrix[genotype_indices].astype(np.float64)
        target = genotype_target.to_numpy(dtype=np.float64)
        center = matrix.mean(axis=0)
        standardized = matrix - center
        denominator = np.sqrt(np.sum(standardized**2, axis=0) * np.sum((target - target.mean()) ** 2))
        score = np.divide(
            np.abs(standardized.T @ (target - target.mean())),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-12,
        )
        count = min(int(self.n_markers), len(score))
        self.marker_indices_ = np.argpartition(score, -count)[-count:]
        self.marker_center_ = data.genotype_matrix[genotype_indices][:, self.marker_indices_].mean(axis=0)
        self.marker_scale_ = data.genotype_matrix[genotype_indices][:, self.marker_indices_].std(axis=0)
        self.marker_scale_[self.marker_scale_ < 1e-8] = 1.0
        self.inputs_ = ProjectedInputs(2, self.environment_components, self.random_state).fit(
            data, indices
        )
        design = self._design(data, indices)
        self.regressor_ = YieldHeadRegressor(
            self.head,
            params=dict(self.head_params or {}),
            random_state=self.random_state,
            device=self.device,
        ).fit(design, _yield(data, indices))
        return self

    def _design(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        rows = data.records.loc[indices]
        marker = data.genotype_matrix[
            rows["genotype_index"].to_numpy(dtype=int)[:, None], self.marker_indices_[None, :]
        ].astype(np.float32)
        marker = (marker - self.marker_center_) / self.marker_scale_
        _, environment = self.inputs_.rows(data, indices)
        if not hasattr(self, "environment_scaler_"):
            self.environment_scaler_ = StandardScaler().fit(environment)
        environment = self.environment_scaler_.transform(environment)
        interaction = np.einsum("bi,bj->bij", marker, environment).reshape(len(marker), -1)
        return np.concatenate([marker, environment, interaction], axis=1).astype(np.float32)

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        return self.regressor_.predict(self._design(data, indices))


class _GEFormerStyleNetwork:
    def __new__(cls, genotype_features: int, environment_features: int, width: int, heads: int, dropout: float):
        import torch
        import torch.nn as nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.genotype = nn.Sequential(
                    nn.Linear(genotype_features, width * 2), nn.GELU(), nn.Dropout(dropout)
                )
                self.environment = nn.Sequential(
                    nn.Conv1d(environment_features, width, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
                self.gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
                self.head = nn.Sequential(
                    nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
                )

            def forward(self, genotype, environment, mask):
                left, right = self.genotype(genotype).chunk(2, dim=1)
                genotype_embedding = left * nn.functional.sigmoid(right)
                encoded = self.environment(environment.transpose(1, 2)).transpose(1, 2)
                key_padding = ~mask.bool()
                attended, _ = self.attention(encoded, encoded, encoded, key_padding_mask=key_padding)
                weights = mask.unsqueeze(-1).to(attended.dtype)
                environment_embedding = (attended * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                gate = self.gate(torch.cat([genotype_embedding, environment_embedding], dim=1))
                fused = gate * genotype_embedding + (1.0 - gate) * environment_embedding
                return self.head(fused).squeeze(1)

        return Network()


@dataclass
class GEFormerStyle:
    """Compact gMLP-temporal-attention GxE baseline inspired by GEFormer.

    It is intentionally labelled ``GEFormer-style`` because the official GEFormer
    release is a Python-3.8 compiled package and cannot be faithfully executed in
    this Python-3.10 project without a separate runtime.
    """

    genotype_components: int = 64
    width: int = 96
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 120
    patience: int = 15
    random_state: int = 2026
    device: str = "auto"

    def fit(self, data: GPACEData, indices: np.ndarray) -> "GEFormerStyle":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device_ = torch.device(
            "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        )
        self.inputs_ = ProjectedInputs(self.genotype_components, 2, self.random_state).fit(data, indices)
        genotype, _ = self.inputs_.rows(data, indices)
        genotype = StandardScaler().fit_transform(genotype).astype(np.float32)
        self.genotype_scaler_ = StandardScaler().fit(self.inputs_.rows(data, indices)[0])
        environment, mask = self._environment_tensor(data, indices)
        target = _yield(data, indices).astype(np.float32)
        self.target_mean_ = float(target.mean())
        self.target_scale_ = float(target.std()) or 1.0
        target = (target - self.target_mean_) / self.target_scale_
        groups = data.records.loc[indices, "Experiment"].to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=min(0.2, max(1 / len(np.unique(groups)), 0.1)), random_state=self.random_state)
        train_pos, validation_pos = next(splitter.split(genotype, groups=groups))
        self.model_ = _GEFormerStyleNetwork(
            genotype.shape[1], environment.shape[-1], self.width, self.heads, self.dropout
        ).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(genotype[train_pos]), torch.from_numpy(environment[train_pos]),
                torch.from_numpy(mask[train_pos]), torch.from_numpy(target[train_pos]),
            ), batch_size=min(self.batch_size, len(train_pos)), shuffle=True
        )
        best_state = None
        best_loss = float("inf")
        stalled = 0
        for _ in range(self.epochs):
            self.model_.train()
            for g, e, m, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model_(g.to(self.device_), e.to(self.device_), m.to(self.device_))
                loss = loss_fn(prediction, y.to(self.device_))
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                validation_prediction = self.model_(
                    torch.from_numpy(genotype[validation_pos]).to(self.device_),
                    torch.from_numpy(environment[validation_pos]).to(self.device_),
                    torch.from_numpy(mask[validation_pos]).to(self.device_),
                )
                validation_loss = float(loss_fn(validation_prediction, torch.from_numpy(target[validation_pos]).to(self.device_)).item())
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
                stalled = 0
            else:
                stalled += 1
                if stalled >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def _environment_tensor(self, data: GPACEData, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        env_indices = data.records.loc[indices, "environment_index"].to_numpy(dtype=int)
        sequence = data.environment_sequences[env_indices]
        if not hasattr(self, "environment_projector_"):
            environment_indices = np.sort(data.records.loc[indices, "environment_index"].unique())
            from .features import EnvironmentProjector
            self.environment_projector_ = EnvironmentProjector(2, self.random_state).fit(
                data.environment_sequences[environment_indices]
            )
        standardized = self.environment_projector_.standardized_sequences(sequence)
        mask = np.isfinite(sequence).any(axis=-1)
        return standardized.astype(np.float32), mask.astype(np.bool_)

    def predict(self, data: GPACEData, indices: np.ndarray) -> np.ndarray:
        import torch

        genotype, _ = self.inputs_.rows(data, indices)
        genotype = self.genotype_scaler_.transform(genotype).astype(np.float32)
        environment, mask = self._environment_tensor(data, indices)
        with torch.no_grad():
            prediction = self.model_(
                torch.from_numpy(genotype).to(self.device_),
                torch.from_numpy(environment).to(self.device_),
                torch.from_numpy(mask).to(self.device_),
            ).cpu().numpy()
        return prediction * self.target_scale_ + self.target_mean_


def make_published_benchmark(
    name: str, random_state: int, device: str = "auto", **params: Any
):
    if name == "ec_gblup":
        return ECGBLUP(random_state=random_state, **params)
    if name == "reaction_norm_kernel":
        return KernelReactionNorm(random_state=random_state, **params)
    if name == "reaction_norm_rff":
        return RFFReactionNorm(random_state=random_state, **params)
    if name == "ceris_jgra":
        return CERISJGRA(random_state=random_state, **params)
    if name == "autogs_style":
        return AutoGSStyle(random_state=random_state, device=device, **params)
    if name == "geformer_style":
        return GEFormerStyle(random_state=random_state, device=device, **params)
    raise ValueError(f"Unknown published benchmark: {name}")


DEFAULT_BENCHMARK_CONFIGS: dict[str, dict[str, Any]] = {
    "ec_gblup": {
        "genotype_components": 64,
        "environment_components": 12,
        "genotype_alpha": 10.0,
        "environment_alpha": 1.0,
        "interaction_alpha": 30.0,
    },
    "reaction_norm_kernel": {
        "genotype_components": 32,
        "environment_components": 12,
        "alpha": 1.0,
        "genotype_weight": 1.0,
        "environment_weight": 1.0,
        "interaction_weight": 1.0,
    },
    "ceris_jgra": {"genotype_components": 32, "alpha": 10.0, "min_window": 7, "max_window": 35, "window_step": 7},
    "autogs_style": {
        "n_markers": 512,
        "environment_components": 12,
        "head": "catboost",
        "head_params": {"iterations": 800, "depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 5.0},
    },
    "geformer_style": {
        "genotype_components": 64,
        "width": 96,
        "heads": 4,
        "dropout": 0.1,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "epochs": 120,
        "patience": 15,
    },
}
