from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .data import GPACEData


@dataclass
class GenotypeProjector:
    n_components: int = 32
    random_state: int = 2026

    def fit(self, matrix: np.ndarray) -> "GenotypeProjector":
        train = np.asarray(matrix, dtype=np.float64)
        allele_frequency = train.mean(axis=0) / 2.0
        scale = np.sqrt(2.0 * allele_frequency * (1.0 - allele_frequency))
        self.valid_markers_ = scale > 1e-8
        self.center_ = 2.0 * allele_frequency[self.valid_markers_]
        self.scale_ = scale[self.valid_markers_]
        standardized = (train[:, self.valid_markers_] - self.center_) / self.scale_
        components = min(self.n_components, len(train) - 1, standardized.shape[1])
        if components < 1:
            raise ValueError("Not enough genotypes to fit genomic PCA")
        self.pca_ = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=self.random_state,
        ).fit(standardized)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)[:, self.valid_markers_]
        values = (values - self.center_) / self.scale_
        return self.pca_.transform(values).astype(np.float32)


@dataclass
class EnvironmentProjector:
    n_components: int = 12
    random_state: int = 2026

    def fit(self, sequences: np.ndarray) -> "EnvironmentProjector":
        train = np.asarray(sequences, dtype=np.float64)
        valid_days = np.isfinite(train).any(axis=-1)
        flattened_days = train.reshape(-1, train.shape[-1])
        self.median_ = np.nanmedian(flattened_days, axis=0)
        if np.isnan(self.median_).any():
            bad = np.flatnonzero(np.isnan(self.median_)).tolist()
            raise ValueError(
                f"Daily variables are entirely missing in training environments: {bad}"
            )
        filled = np.where(np.isnan(train), self.median_[None, None, :], train)
        flattened_days = filled.reshape(-1, filled.shape[-1])
        self.mean_ = flattened_days.mean(axis=0)
        self.scale_ = flattened_days.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (filled - self.mean_[None, None, :]) / self.scale_[None, None, :]
        standardized[~valid_days] = 0.0
        flattened = standardized.reshape(len(standardized), -1)
        components = min(self.n_components, len(train) - 1, flattened.shape[1])
        if components < 1:
            raise ValueError("Not enough environments to fit environmental PCA")
        self.pca_ = PCA(n_components=components, svd_solver="full").fit(flattened)
        return self

    def standardized_sequences(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=np.float64)
        valid_days = np.isfinite(values).any(axis=-1)
        filled = np.where(np.isnan(values), self.median_[None, None, :], values)
        standardized = (filled - self.mean_[None, None, :]) / self.scale_[None, None, :]
        standardized[~valid_days] = 0.0
        return standardized.astype(np.float32)

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        standardized = self.standardized_sequences(sequences)
        flattened = standardized.reshape(len(standardized), -1)
        return self.pca_.transform(flattened).astype(np.float32)


@dataclass
class ProjectedInputs:
    genotype_components: int = 32
    environment_components: int = 12
    random_state: int = 2026

    def fit(self, data: GPACEData, row_indices: np.ndarray) -> "ProjectedInputs":
        rows = data.records.loc[row_indices]
        genotype_indices = np.sort(rows["genotype_index"].unique())
        environment_indices = np.sort(rows["environment_index"].unique())
        self.genotype_projector_ = GenotypeProjector(
            self.genotype_components, self.random_state
        ).fit(data.genotype_matrix[genotype_indices])
        self.environment_projector_ = EnvironmentProjector(
            self.environment_components, self.random_state
        ).fit(data.environment_sequences[environment_indices])
        self.genotype_embedding_ = self.genotype_projector_.transform(data.genotype_matrix)
        self.environment_embedding_ = self.environment_projector_.transform(
            data.environment_sequences
        )
        return self

    def rows(self, data: GPACEData, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = data.records.loc[row_indices]
        genotype = self.genotype_embedding_[rows["genotype_index"].to_numpy(dtype=int)]
        environment = self.environment_embedding_[rows["environment_index"].to_numpy(dtype=int)]
        return genotype, environment


def interaction_design(genotype: np.ndarray, environment: np.ndarray) -> np.ndarray:
    interaction = np.einsum("bi,bj->bij", genotype, environment).reshape(len(genotype), -1)
    return np.concatenate([genotype, environment, interaction], axis=1)


def phenology_design(genotype: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Stage-conditioned environmental features without duplicating genotype PCs."""

    interaction = np.einsum("bi,bj->bij", genotype, exposures).reshape(len(genotype), -1)
    return np.concatenate([exposures, interaction], axis=1)


def additive_design(genotype: np.ndarray, environment: np.ndarray) -> np.ndarray:
    return np.concatenate([genotype, environment], axis=1)
