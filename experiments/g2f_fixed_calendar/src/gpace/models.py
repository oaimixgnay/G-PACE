from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data import GPACEData, EXPOSURE_COLUMNS
from .features import (
    ProjectedInputs,
    additive_design,
    interaction_design,
    phenology_design,
)


class _ScaledRidge:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ScaledRidge":
        self.x_scaler_ = StandardScaler().fit(x)
        self.y_scaler_ = StandardScaler().fit(y)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(self.x_scaler_.transform(x), self.y_scaler_.transform(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model_.predict(self.x_scaler_.transform(x))
        return self.y_scaler_.inverse_transform(prediction)


class YieldHeadRegressor:
    """A common scaled interface for the selectable yield regression heads."""

    def __init__(
        self,
        kind: str,
        params: dict[str, Any] | None = None,
        random_state: int = 2026,
        device: str = "auto",
        scale_features: bool = True,
    ):
        self.kind = kind
        self.params = dict(params or {})
        self.random_state = random_state
        self.device = device
        self.scale_features = scale_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "YieldHeadRegressor":
        self.x_scaler_ = StandardScaler().fit(x) if self.scale_features else None
        self.y_scaler_ = StandardScaler().fit(np.asarray(y).reshape(-1, 1))
        x_scaled = self.x_scaler_.transform(x) if self.x_scaler_ is not None else x
        y_scaled = self.y_scaler_.transform(np.asarray(y).reshape(-1, 1)).ravel()

        if self.kind == "ridge":
            self.model_ = Ridge(alpha=float(self.params.get("alpha", 10.0)))
        elif self.kind == "krr":
            gamma = self.params.get("gamma")
            if gamma is None:
                gamma = 1.0 / max(1, x_scaled.shape[1])
            self.model_ = KernelRidge(
                alpha=float(self.params.get("alpha", 1.0)),
                gamma=float(gamma),
                kernel="rbf",
            )
        elif self.kind == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:
                raise ImportError(
                    "CatBoost is required for the catboost yield head. "
                    "Install the search extras with: pip install -e '.[search]'"
                ) from exc
            task_type = self.params.get("task_type")
            if task_type is None:
                device = self.device
                if device == "auto":
                    try:
                        import torch

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"
                task_type = "GPU" if device.startswith("cuda") else "CPU"
            options = {
                "iterations": 800,
                "depth": 6,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
                "loss_function": "RMSE",
                "random_seed": self.random_state,
                "verbose": False,
                "allow_writing_files": False,
                "task_type": task_type,
            }
            options.update(self.params)
            self.model_ = CatBoostRegressor(**options)
        else:
            raise ValueError(f"Unknown yield head: {self.kind}")

        self.model_.fit(x_scaled, y_scaled)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = self.x_scaler_.transform(x) if self.x_scaler_ is not None else x
        prediction = np.asarray(self.model_.predict(values)).reshape(-1, 1)
        return self.y_scaler_.inverse_transform(prediction).ravel()


class EnvironmentMeanRegressor:
    """Environment mean model with optional leave-one-environment calibration."""

    def __init__(
        self,
        kind: str = "ridge",
        alpha: float = 10.0,
        gamma: float | None = None,
        calibration_strength: float = 0.0,
    ):
        self.kind = kind
        self.alpha = float(alpha)
        self.gamma = gamma
        self.calibration_strength = float(calibration_strength)

    def _model(self) -> YieldHeadRegressor:
        params: dict[str, float] = {"alpha": self.alpha}
        if self.gamma is not None:
            params["gamma"] = float(self.gamma)
        return YieldHeadRegressor(self.kind, params=params, device="cpu")

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnvironmentMeanRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if len(y) < 3:
            raise ValueError("At least three environments are required")
        self.calibration_slope_ = 1.0
        self.calibration_intercept_ = 0.0
        if self.calibration_strength > 0.0:
            loo = np.empty(len(y), dtype=np.float64)
            for held_out in range(len(y)):
                keep = np.arange(len(y)) != held_out
                loo[held_out] = self._model().fit(x[keep], y[keep]).predict(
                    x[held_out : held_out + 1]
                )[0]
            if np.std(loo) > 1e-8:
                slope, intercept = np.polyfit(loo, y, 1)
                slope = float(np.clip(slope, 0.0, 3.0))
                strength = float(np.clip(self.calibration_strength, 0.0, 1.0))
                self.calibration_slope_ = 1.0 + strength * (slope - 1.0)
                self.calibration_intercept_ = strength * float(intercept)
        self.model_ = self._model().fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = self.model_.predict(np.asarray(x, dtype=np.float32))
        return self.calibration_intercept_ + self.calibration_slope_ * raw


class GenomicBLUPRegressor:
    """Dual GBLUP on unique genotypes with analytic leave-one-out predictions."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def fit(self, matrix: np.ndarray, target: np.ndarray) -> "GenomicBLUPRegressor":
        train = np.asarray(matrix, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        frequency = train.mean(axis=0) / 2.0
        scale = np.sqrt(2.0 * frequency * (1.0 - frequency))
        self.valid_markers_ = scale > 1e-8
        self.center_ = 2.0 * frequency[self.valid_markers_]
        self.scale_ = scale[self.valid_markers_]
        standardized = (
            train[:, self.valid_markers_] - self.center_[None, :]
        ) / self.scale_[None, :]
        self.standardized_train_ = standardized.astype(np.float32)
        marker_count = self.standardized_train_.shape[1]
        relationship = (
            self.standardized_train_ @ self.standardized_train_.T
        ) / max(1, marker_count)
        eigenvalues, eigenvectors = np.linalg.eigh(relationship.astype(np.float64))
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        inverse = 1.0 / (eigenvalues + self.alpha)
        self.dual_coef_ = eigenvectors @ (inverse * (eigenvectors.T @ target))
        leverage = (eigenvectors**2) @ (eigenvalues * inverse)
        fitted = relationship @ self.dual_coef_
        denominator = np.maximum(1e-6, 1.0 - leverage)
        self.loo_prediction_ = (fitted - leverage * target) / denominator
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)[:, self.valid_markers_]
        standardized = (values - self.center_[None, :]) / self.scale_[None, :]
        relationship = (
            standardized.astype(np.float32) @ self.standardized_train_.T
        ) / max(1, self.standardized_train_.shape[1])
        return np.asarray(relationship @ self.dual_coef_, dtype=np.float64)


@dataclass
class RidgeExposureModel:
    genotype_components: int = 32
    environment_components: int = 12
    alpha: float = 10.0
    random_state: int = 2026

    def fit(self, data: GPACEData, row_indices: np.ndarray) -> "RidgeExposureModel":
        self.inputs_ = ProjectedInputs(
            self.genotype_components,
            self.environment_components,
            self.random_state,
        ).fit(data, row_indices)
        genotype, environment = self.inputs_.rows(data, row_indices)
        x = interaction_design(genotype, environment)
        y = data.records.loc[row_indices, EXPOSURE_COLUMNS].to_numpy(dtype=np.float64)
        self.regressor_ = _ScaledRidge(self.alpha).fit(x, y)
        return self

    def predict(self, data: GPACEData, row_indices: np.ndarray) -> np.ndarray:
        genotype, environment = self.inputs_.rows(data, row_indices)
        return self.regressor_.predict(interaction_design(genotype, environment))


class _TCNNetwork:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class ResidualBlock(nn.Module):
            def __init__(
                self, width: int, dilation: int, kernel_size: int, dropout: float
            ):
                super().__init__()
                padding = dilation * (kernel_size - 1) // 2
                self.block = nn.Sequential(
                    nn.Conv1d(
                        width,
                        width,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(
                        width,
                        width,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.norm = nn.GroupNorm(1, width)

            def forward(self, x):
                return self.norm(x + self.block(x))

        class Network(nn.Module):
            def __init__(
                self,
                env_features,
                genotype_features,
                outputs,
                width,
                depth,
                kernel_size,
                dropout,
            ):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(env_features + 3, width, kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                )
                self.genotype = nn.Sequential(
                    nn.Linear(genotype_features, width), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks = nn.ModuleList(
                    [
                        ResidualBlock(width, 2**block, kernel_size, dropout)
                        for block in range(depth)
                    ]
                )
                self.film = nn.ModuleList(
                    [nn.Linear(width, width * 2) for _ in range(depth)]
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Sequential(
                    nn.Linear(width * 2, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, outputs),
                )

            def forward(self, genotype, environment, mask=None):
                if mask is None:
                    mask = torch.ones(
                        environment.shape[:2], device=environment.device, dtype=environment.dtype
                    )
                else:
                    mask = mask.to(dtype=environment.dtype)
                days = torch.linspace(
                    -1.0,
                    1.0,
                    environment.shape[1],
                    device=environment.device,
                    dtype=environment.dtype,
                )
                position = torch.stack(
                    [days, torch.sin(torch.pi * days), torch.cos(torch.pi * days)], dim=0
                )
                position = position.unsqueeze(0).expand(environment.shape[0], -1, -1)
                valid = mask.unsqueeze(1)
                position = position * valid
                env = self.stem(torch.cat([environment.transpose(1, 2), position], dim=1))
                env = env * valid
                geno = self.genotype(genotype)
                for block, film in zip(self.blocks, self.film):
                    env = block(env)
                    gamma, beta = film(geno).chunk(2, dim=1)
                    env = env * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
                    env = env * valid
                env = (env * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
                return self.head(torch.cat([geno, env], dim=1))

        return Network(*args, **kwargs)


@dataclass
class TCNExposureModel:
    genotype_components: int = 32
    width: int = 64
    depth: int = 3
    kernel_size: int = 5
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    patience: int = 12
    validation_fraction: float = 0.15
    amp: bool = True
    refit_full: bool = True
    verbose: bool = False
    random_state: int = 2026
    device: str = "auto"

    def fit(self, data: GPACEData, row_indices: np.ndarray) -> "TCNExposureModel":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device_ = torch.device(
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )

        self.inputs_ = ProjectedInputs(
            genotype_components=self.genotype_components,
            environment_components=2,
            random_state=self.random_state,
        ).fit(data, row_indices)
        rows = data.records.loc[row_indices]
        genotype_all = self.inputs_.genotype_embedding_[rows["genotype_index"].to_numpy(int)]
        standardized_environments = self.inputs_.environment_projector_.standardized_sequences(
            data.environment_sequences
        )
        environment_all = standardized_environments[rows["environment_index"].to_numpy(int)]
        environment_lengths = data.environment_sequence_lengths[
            rows["environment_index"].to_numpy(int)
        ]
        environment_mask = (
            np.arange(environment_all.shape[1])[None, :] < environment_lengths[:, None]
        ).astype(np.float32)
        target_all = rows.loc[:, EXPOSURE_COLUMNS].to_numpy(dtype=np.float32)
        self.target_scaler_ = StandardScaler().fit(target_all)
        target_all = self.target_scaler_.transform(target_all).astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        genotypes = rows["Variety_ID"].drop_duplicates().to_numpy(dtype=str)
        environments = rows["Experiment"].drop_duplicates().to_numpy(dtype=str)
        held_genotypes = set(
            rng.choice(
                genotypes,
                size=max(1, int(round(len(genotypes) * self.validation_fraction))),
                replace=False,
            )
        )
        held_environments = set(
            rng.choice(
                environments,
                size=max(1, int(round(len(environments) * self.validation_fraction))),
                replace=False,
            )
        )
        valid_mask = rows["Variety_ID"].isin(held_genotypes).to_numpy() | rows[
            "Experiment"
        ].isin(held_environments).to_numpy()
        fit_pos = np.flatnonzero(~valid_mask)
        valid_pos = np.flatnonzero(valid_mask)

        def dataset(position: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.from_numpy(genotype_all[position]),
                torch.from_numpy(environment_all[position]),
                torch.from_numpy(environment_mask[position]),
                torch.from_numpy(target_all[position]),
            )

        generator = torch.Generator().manual_seed(self.random_state)
        train_loader = DataLoader(
            dataset(fit_pos),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=self.device_.type == "cuda",
        )
        valid_loader = DataLoader(
            dataset(valid_pos),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device_.type == "cuda",
        )
        self.network_ = _TCNNetwork(
            environment_all.shape[-1],
            genotype_all.shape[-1],
            target_all.shape[-1],
            self.width,
            self.depth,
            self.kernel_size,
            self.dropout,
        ).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_function = torch.nn.SmoothL1Loss()
        best_loss = np.inf
        best_state = None
        best_epoch = 0
        stale = 0
        amp_enabled = bool(self.amp and self.device_.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for epoch in range(self.epochs):
            self.network_.train()
            for genotype, environment, mask, target in train_loader:
                genotype = genotype.to(self.device_, non_blocking=True)
                environment = environment.to(self.device_, non_blocking=True)
                mask = mask.to(self.device_, non_blocking=True)
                target = target.to(self.device_, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device_.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    loss = loss_function(self.network_(genotype, environment, mask), target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            self.network_.eval()
            losses = []
            with torch.no_grad():
                for genotype, environment, mask, target in valid_loader:
                    with torch.autocast(
                        device_type=self.device_.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        prediction = self.network_(
                            genotype.to(self.device_, non_blocking=True),
                            environment.to(self.device_, non_blocking=True),
                            mask.to(self.device_, non_blocking=True),
                        )
                        loss = loss_function(
                            prediction, target.to(self.device_, non_blocking=True)
                        )
                    losses.append(loss.item())
            valid_loss = float(np.mean(losses))
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"[TCN] seed={self.random_state} epoch={epoch + 1}/{self.epochs} validation_loss={valid_loss:.6f}", flush=True)
            if valid_loss < best_loss - 1e-5:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network_.state_dict().items()
                }
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("TCN training did not produce a valid checkpoint")
        self.network_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.best_validation_loss_ = float(best_loss)

        if self.refit_full:
            if self.verbose:
                print(f"[TCN] refit all {len(rows)} training rows for {best_epoch} epochs", flush=True)
            torch.manual_seed(self.random_state)
            full_loader = DataLoader(
                dataset(np.arange(len(rows))),
                batch_size=self.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.random_state),
                pin_memory=self.device_.type == "cuda",
            )
            self.network_ = _TCNNetwork(
                environment_all.shape[-1],
                genotype_all.shape[-1],
                target_all.shape[-1],
                self.width,
                self.depth,
                self.kernel_size,
                self.dropout,
            ).to(self.device_)
            optimizer = torch.optim.AdamW(
                self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            for _ in range(best_epoch):
                if self.verbose and (_ == 0 or (_ + 1) % 10 == 0):
                    print(f"[TCN] full-refit epoch={_ + 1}/{best_epoch}", flush=True)
                self.network_.train()
                for genotype, environment, mask, target in full_loader:
                    genotype = genotype.to(self.device_, non_blocking=True)
                    environment = environment.to(self.device_, non_blocking=True)
                    mask = mask.to(self.device_, non_blocking=True)
                    target = target.to(self.device_, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=self.device_.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        loss = loss_function(self.network_(genotype, environment, mask), target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        return self

    def predict(self, data: GPACEData, row_indices: np.ndarray) -> np.ndarray:
        import torch

        rows = data.records.loc[row_indices]
        genotype = self.inputs_.genotype_embedding_[rows["genotype_index"].to_numpy(int)]
        environments = self.inputs_.environment_projector_.standardized_sequences(
            data.environment_sequences
        )[rows["environment_index"].to_numpy(int)]
        lengths = data.environment_sequence_lengths[rows["environment_index"].to_numpy(int)]
        mask = (
            np.arange(environments.shape[1])[None, :] < lengths[:, None]
        ).astype(np.float32)
        predictions = []
        self.network_.eval()
        with torch.no_grad():
            for start in range(0, len(rows), self.batch_size):
                stop = start + self.batch_size
                batch = self.network_(
                    torch.from_numpy(genotype[start:stop]).to(self.device_),
                    torch.from_numpy(environments[start:stop]).to(self.device_),
                    torch.from_numpy(mask[start:stop]).to(self.device_),
                )
                predictions.append(batch.cpu().numpy())
        standardized = np.concatenate(predictions, axis=0)
        return self.target_scaler_.inverse_transform(standardized)


def make_exposure_model(kind: str, random_state: int, **kwargs):
    if kind == "ridge":
        allowed = {"genotype_components", "environment_components", "alpha"}
        return RidgeExposureModel(
            random_state=random_state, **{k: v for k, v in kwargs.items() if k in allowed}
        )
    if kind in {"tcn", "tcn_film"}:
        allowed = {
            "genotype_components",
            "width",
            "depth",
            "kernel_size",
            "dropout",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "patience",
            "validation_fraction",
            "amp",
            "refit_full",
            "verbose",
            "device",
        }
        return TCNExposureModel(
            random_state=random_state, **{k: v for k, v in kwargs.items() if k in allowed}
        )
    raise ValueError(f"Unknown exposure model: {kind}")


def crossfit_exposures(
    data: GPACEData,
    row_indices: np.ndarray,
    kind: str,
    folds: int,
    random_state: int,
    model_kwargs: dict | None = None,
) -> np.ndarray:
    model_kwargs = model_kwargs or {}
    rows = data.records.loc[row_indices]
    predictions = np.zeros((len(rows), len(EXPOSURE_COLUMNS)), dtype=np.float64)
    prediction_counts = np.zeros(len(rows), dtype=int)

    for axis, group_column in enumerate(("Variety_ID", "Experiment")):
        groups = rows[group_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if folds > len(unique_groups):
            raise ValueError(
                f"crossfit folds ({folds}) exceed unique {group_column} groups ({len(unique_groups)})"
            )
        rng = np.random.default_rng(random_state + axis)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        fold_groups = np.array_split(shuffled_groups, folds)
        for fold, held_groups in enumerate(fold_groups):
            predict_mask = np.isin(groups, held_groups)
            predict_pos = np.flatnonzero(predict_mask)
            fit_pos = np.flatnonzero(~predict_mask)
            model = make_exposure_model(
                kind,
                random_state=random_state + axis * 100 + fold,
                **model_kwargs,
            )
            fit_indices = rows.index.to_numpy()[fit_pos]
            predict_indices = rows.index.to_numpy()[predict_pos]
            predictions[predict_pos] += model.fit(data, fit_indices).predict(data, predict_indices)
            prediction_counts[predict_pos] += 1

    if not np.all(prediction_counts == 2):
        raise RuntimeError(
            "Every training record must receive one genotype-held-out and one environment-held-out prediction"
        )
    return predictions / prediction_counts[:, None]


@dataclass
class YieldModel:
    mode: Literal[
        "g_only",
        "e_only",
        "g_plus_e",
        "e2e",
        "gpace",
        "oracle",
        "hybrid",
        "hybrid_oracle",
        "mean_residual_hybrid",
        "v3_mean_residual",
    ]
    genotype_components: int = 32
    environment_components: int = 12
    alpha: float = 10.0
    head: str = "ridge"
    head_params: dict[str, Any] | None = None
    random_state: int = 2026
    device: str = "auto"

    def fit(
        self,
        data: GPACEData,
        row_indices: np.ndarray,
        exposures: np.ndarray | None = None,
    ) -> "YieldModel":
        self.inputs_ = ProjectedInputs(
            self.genotype_components,
            self.environment_components,
            self.random_state,
        ).fit(data, row_indices)
        self.exposure_scaler_ = None
        self.direct_scaler_ = None
        self.phenology_scaler_ = None
        self.phenology_weight_ = 1.0
        uses_hybrid_scaling = self.mode in {
            "hybrid",
            "hybrid_oracle",
            "mean_residual_hybrid",
            "v3_mean_residual",
        }
        x = self._design(data, row_indices, exposures, fit=True)
        y = data.records.loc[row_indices, "grain.yield"].to_numpy(dtype=np.float64)
        params = dict(self.head_params or {})
        mean_alpha = float(params.pop("mean_alpha", 10.0))
        mean_head = str(params.pop("mean_head", "ridge"))
        mean_gamma = params.pop("mean_gamma", None)
        mean_calibration_strength = float(
            params.pop("mean_calibration_strength", 0.0)
        )
        self.residual_weight_ = float(params.pop("residual_weight", 1.0))
        self.gblup_weight_ = float(params.pop("gblup_weight", 0.0))
        gblup_alpha = float(params.pop("gblup_alpha", 1.0))
        params.pop("yield_genotype_components", None)
        params.pop("phenology_weight", None)
        params.pop("yield_mode", None)
        if self.head == "ridge":
            params.setdefault("alpha", self.alpha)
        if self.mode in {"mean_residual_hybrid", "v3_mean_residual"}:
            # Keep environment mean prediction separate from genotype deviations.
            _, environment = self.inputs_.rows(data, row_indices)
            experiments = data.records.loc[row_indices, "Experiment"].to_numpy()
            environment_means = pd.Series(y).groupby(experiments, sort=False).transform("mean")
            unique_positions = (
                pd.Series(experiments).drop_duplicates(keep="first").index.to_numpy(dtype=int)
            )
            if self.mode == "v3_mean_residual":
                self.mean_regressor_ = EnvironmentMeanRegressor(
                    kind=mean_head,
                    alpha=mean_alpha,
                    gamma=mean_gamma,
                    calibration_strength=mean_calibration_strength,
                ).fit(
                    environment[unique_positions],
                    environment_means.to_numpy(dtype=np.float64)[unique_positions],
                )
            else:
                self.mean_regressor_ = YieldHeadRegressor(
                    "ridge",
                    params={"alpha": mean_alpha},
                    random_state=self.random_state,
                    device=self.device,
                ).fit(
                    environment[unique_positions],
                    environment_means.to_numpy(dtype=np.float64)[unique_positions],
                )
            residual_target = y - environment_means.to_numpy(dtype=np.float64)
            self.gblup_ = None
            if self.mode == "v3_mean_residual" and self.gblup_weight_ > 0.0:
                genotype_indices = data.records.loc[
                    row_indices, "genotype_index"
                ].to_numpy(dtype=int)
                genotype_effects = (
                    pd.Series(residual_target)
                    .groupby(genotype_indices, sort=False)
                    .transform("mean")
                    .to_numpy(dtype=np.float64)
                )
                unique_genotype_positions = (
                    pd.Series(genotype_indices)
                    .drop_duplicates(keep="first")
                    .index.to_numpy(dtype=int)
                )
                unique_genotypes = genotype_indices[unique_genotype_positions]
                self.gblup_ = GenomicBLUPRegressor(alpha=gblup_alpha).fit(
                    data.genotype_matrix[unique_genotypes],
                    genotype_effects[unique_genotype_positions],
                )
                loo_lookup = dict(zip(unique_genotypes, self.gblup_.loo_prediction_))
                loo_prediction = np.asarray(
                    [loo_lookup[index] for index in genotype_indices], dtype=np.float64
                )
                residual_target = residual_target - self.gblup_weight_ * loo_prediction
            self.regressor_ = YieldHeadRegressor(
                self.head,
                params=params,
                random_state=self.random_state,
                device=self.device,
                scale_features=False,
            ).fit(x, residual_target)
        else:
            self.regressor_ = YieldHeadRegressor(
                self.head,
                params=params,
                random_state=self.random_state,
                device=self.device,
                scale_features=not uses_hybrid_scaling,
            ).fit(x, y)
        return self

    def _design(
        self,
        data: GPACEData,
        row_indices: np.ndarray,
        exposures: np.ndarray | None,
        fit: bool,
    ) -> np.ndarray:
        genotype, environment = self.inputs_.rows(data, row_indices)
        if self.mode == "g_only":
            return genotype
        if self.mode == "e_only":
            return environment
        if self.mode == "g_plus_e":
            return additive_design(genotype, environment)
        if self.mode == "e2e":
            return interaction_design(genotype, environment)
        if self.mode in {"gpace", "oracle"}:
            if exposures is None:
                raise ValueError(f"{self.mode} requires exposure features")
            if fit:
                self.exposure_scaler_ = StandardScaler().fit(exposures)
            standardized = self.exposure_scaler_.transform(exposures).astype(np.float32)
            return interaction_design(genotype, standardized)
        if self.mode in {
            "hybrid",
            "hybrid_oracle",
            "mean_residual_hybrid",
            "v3_mean_residual",
        }:
            if exposures is None:
                raise ValueError(f"{self.mode} requires exposure features")
            direct = interaction_design(genotype, environment)
            if fit:
                self.direct_scaler_ = StandardScaler().fit(direct)
                self.exposure_scaler_ = StandardScaler().fit(exposures)
                standardized_exposure = self.exposure_scaler_.transform(exposures)
                phenology = phenology_design(genotype, standardized_exposure)
                self.phenology_scaler_ = StandardScaler().fit(phenology)
                self.phenology_weight_ = float(
                    (self.head_params or {}).get("phenology_weight", 1.0)
                )
            else:
                standardized_exposure = self.exposure_scaler_.transform(exposures)
                phenology = phenology_design(genotype, standardized_exposure)
            return np.concatenate(
                [
                    self.direct_scaler_.transform(direct),
                    self.phenology_weight_ * self.phenology_scaler_.transform(phenology),
                ],
                axis=1,
            ).astype(np.float32)
        raise ValueError(self.mode)

    def predict(
        self,
        data: GPACEData,
        row_indices: np.ndarray,
        exposures: np.ndarray | None = None,
    ) -> np.ndarray:
        x = self._design(data, row_indices, exposures, fit=False)
        residual = self.regressor_.predict(x)
        if self.mode not in {"mean_residual_hybrid", "v3_mean_residual"}:
            return residual
        _, environment = self.inputs_.rows(data, row_indices)
        prediction = (
            self.mean_regressor_.predict(environment) + self.residual_weight_ * residual
        )
        if self.mode == "v3_mean_residual" and self.gblup_ is not None:
            genotype_indices = data.records.loc[
                row_indices, "genotype_index"
            ].to_numpy(dtype=int)
            prediction += self.gblup_weight_ * self.gblup_.predict(
                data.genotype_matrix[genotype_indices]
            )
        return prediction


# Backward-compatible name used by early scripts.
YieldRidgeModel = YieldModel
