from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GPACEData


@dataclass(frozen=True)
class OuterSplit:
    split: int
    train: np.ndarray
    cv0: np.ndarray
    cv1: np.ndarray
    cv00: np.ndarray

    def evaluation_sets(self) -> dict[str, np.ndarray]:
        return {
            "cv0": self.cv0,
            "cv1": self.cv1,
            "cv00": self.cv00,
        }


@dataclass(frozen=True)
class InnerSplit:
    repeat: int
    train: np.ndarray
    cv0: np.ndarray
    cv1: np.ndarray
    cv00: np.ndarray

    def evaluation_sets(self) -> dict[str, np.ndarray]:
        return {"cv0": self.cv0, "cv1": self.cv1, "cv00": self.cv00}


@dataclass(frozen=True)
class EnvironmentHoldout:
    """A complete environmental group withheld from model fitting."""

    scheme: str
    group: str
    train: np.ndarray
    test: np.ndarray

    @property
    def scenario(self) -> str:
        return f"{self.scheme}_{self.group}"


_REGION_BY_SITE = {
    "Cam": "western_europe",
    "Gai": "western_europe",
    "Kar": "western_europe",
    "Ner": "western_europe",
    "Cra": "eastern_europe",
    "Deb": "eastern_europe",
    "Mar": "eastern_europe",
    "Mur": "eastern_europe",
    "Gra": "south_america",
}


def make_outer_split(data: GPACEData, split: int) -> OuterSplit:
    if split not in range(1, 11):
        raise ValueError("split must be between 1 and 10")
    column = f"training.set{split}"
    genotype_train = set(
        data.genotype_partitions.loc[
            data.genotype_partitions[column].str.lower().eq("yes"), "Variety_ID"
        ].astype(str)
    )
    environment_train = set(
        data.environment_partitions.loc[
            data.environment_partitions[column].str.lower().eq("yes"), "Experiment"
        ].astype(str)
    )

    frame = data.records
    is_cv = frame["scenario"].eq("cross_validation")
    g_seen = frame["Variety_ID"].isin(genotype_train)
    e_seen = frame["Experiment"].isin(environment_train)

    def idx(mask) -> np.ndarray:
        return frame.index[np.asarray(mask)].to_numpy(dtype=int)

    result = OuterSplit(
        split=split,
        train=idx(is_cv & g_seen & e_seen),
        cv0=idx(is_cv & g_seen & ~e_seen),
        cv1=idx(is_cv & ~g_seen & e_seen),
        cv00=idx(is_cv & ~g_seen & ~e_seen),
    )
    expected = {"train": 4000, "cv0": 1000, "cv1": 920, "cv00": 230}
    for name, size in expected.items():
        actual = len(getattr(result, name))
        if actual != size:
            raise ValueError(f"Split {split} {name}: expected {size}, found {actual}")
    return result


def final_external_sets(data: GPACEData) -> dict[str, np.ndarray]:
    """Return the two independent source-paper external validation datasets.

    This split happens only after model choices are frozen using the ten
    cross-validation partitions. It must not be mixed with an outer CV split.
    """

    frame = data.records
    return {
        "external_g_ne": frame.index[frame["scenario"].eq("external_known_genotype")].to_numpy(
            dtype=int
        ),
        "external_ng_ne": frame.index[frame["scenario"].eq("external_new_genotype")].to_numpy(
            dtype=int
        ),
    }


def make_environment_group_holdouts(
    data: GPACEData,
    scheme: str,
) -> tuple[EnvironmentHoldout, ...]:
    """Hold out complete CV environments by geographic region or stress class.

    These are deliberately fixed-configuration transfer analyses. The supplied
    CV partitions and the two source-paper external datasets remain separate
    evaluation protocols.
    """

    if scheme not in {"region", "stress"}:
        raise ValueError("scheme must be 'region' or 'stress'")
    environments = data.environment_partitions.loc[:, ["Experiment", "EC"]].copy()
    environments["Experiment"] = environments["Experiment"].astype(str)
    if environments["Experiment"].duplicated().any():
        raise ValueError("Environment partition table contains duplicate experiments")

    if scheme == "region":
        sites = environments["Experiment"].str.slice(stop=3)
        missing = sorted(set(sites) - set(_REGION_BY_SITE))
        if missing:
            raise ValueError(f"No region mapping for CV sites: {missing}")
        environments["group"] = sites.map(_REGION_BY_SITE)
    else:
        environments["group"] = environments["EC"].astype(str).str.replace(
            "-", "_", regex=False
        ).str.lower()

    frame = data.records
    is_cv = frame["scenario"].eq("cross_validation")
    groups = dict(zip(environments["Experiment"], environments["group"], strict=True))
    row_groups = frame["Experiment"].map(groups)
    if row_groups.loc[is_cv].isna().any():
        missing = sorted(frame.loc[is_cv & row_groups.isna(), "Experiment"].unique())
        raise ValueError(f"CV records lack a {scheme} group: {missing}")

    result: list[EnvironmentHoldout] = []
    for group in sorted(environments["group"].unique()):
        test_mask = is_cv & row_groups.eq(group)
        train_mask = is_cv & ~row_groups.eq(group)
        test = frame.index[test_mask].to_numpy(dtype=int)
        train = frame.index[train_mask].to_numpy(dtype=int)
        if not len(test) or not len(train):
            raise ValueError(f"{scheme}={group} produced an empty train or test set")
        test_environments = set(frame.loc[test, "Experiment"])
        train_environments = set(frame.loc[train, "Experiment"])
        if test_environments & train_environments:
            raise RuntimeError(f"{scheme}={group} leaks held-out environments into training")
        result.append(EnvironmentHoldout(scheme, str(group), train, test))
    return tuple(result)


def make_cv2_holdout(
    data: GPACEData,
    train_indices: np.ndarray,
    fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    frame = data.records.loc[train_indices]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    target = int(round(len(frame) * fraction))
    holdout_mask = np.zeros(len(frame), dtype=bool)

    genotype_counts = frame["Variety_ID"].value_counts().to_dict()
    environment_counts = frame["Experiment"].value_counts().to_dict()
    current_g = genotype_counts.copy()
    current_e = environment_counts.copy()

    for position in order:
        if int(holdout_mask.sum()) >= target:
            break
        row = frame.iloc[position]
        genotype = row["Variety_ID"]
        environment = row["Experiment"]
        if current_g[genotype] <= 1 or current_e[environment] <= 1:
            continue
        holdout_mask[position] = True
        current_g[genotype] -= 1
        current_e[environment] -= 1

    if int(holdout_mask.sum()) != target:
        raise RuntimeError(
            "Could not construct a CV2 holdout while retaining every genotype and environment"
        )
    train = frame.index[~holdout_mask].to_numpy(dtype=int)
    holdout = frame.index[holdout_mask].to_numpy(dtype=int)
    return train, holdout


def make_inner_splits(
    data: GPACEData,
    train_indices: np.ndarray,
    repeats: int = 2,
    holdout_fraction: float = 0.2,
    seed: int = 2026,
) -> tuple[InnerSplit, ...]:
    """Create training-only G/E checkerboard splits for hyperparameter selection."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    frame = data.records.loc[train_indices]
    genotype_ids = frame["Variety_ID"].drop_duplicates().to_numpy(dtype=str)
    environment_ids = frame["Experiment"].drop_duplicates().to_numpy(dtype=str)
    if len(genotype_ids) < 5 or len(environment_ids) < 5:
        raise ValueError("Too few groups to construct an inner G/E split")

    result: list[InnerSplit] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        held_genotypes = set(
            rng.choice(
                genotype_ids,
                size=max(1, int(round(len(genotype_ids) * holdout_fraction))),
                replace=False,
            )
        )
        held_environments = set(
            rng.choice(
                environment_ids,
                size=max(1, int(round(len(environment_ids) * holdout_fraction))),
                replace=False,
            )
        )
        g_held = frame["Variety_ID"].isin(held_genotypes).to_numpy()
        e_held = frame["Experiment"].isin(held_environments).to_numpy()
        source_indices = frame.index.to_numpy(dtype=int)

        def select(mask: np.ndarray) -> np.ndarray:
            return source_indices[mask]

        split = InnerSplit(
            repeat=repeat,
            train=select(~g_held & ~e_held),
            cv0=select(~g_held & e_held),
            cv1=select(g_held & ~e_held),
            cv00=select(g_held & e_held),
        )
        if any(len(indices) == 0 for indices in split.evaluation_sets().values()):
            raise RuntimeError("An inner validation scenario is empty")
        result.append(split)
    return tuple(result)
