from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass
class StageEncodings:
    t_stage_order: tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4")
    n_stage_order: tuple[str, ...] = ("N0", "N1", "N2", "N3", "Nx", "Np")

    def encode_t(self, value: str | float | int | None) -> int:
        return self._encode(value, self.t_stage_order)

    def encode_n(self, value: str | float | int | None) -> int:
        return self._encode(value, self.n_stage_order)

    @staticmethod
    def _encode(value: str | float | int | None, order: Iterable[str]) -> int:
        categories = {name: idx for idx, name in enumerate(order)}
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return -1
        value = str(value).strip()
        if value in categories:
            return categories[value]
        if value.startswith("T") or value.startswith("N"):
            digits = "".join(ch for ch in value if ch.isdigit())
            fallback = f"{value[0]}{digits}" if digits else value[:2]
            return categories.get(fallback, -1)
        return -1


class ClinicalPreprocessor:
    def __init__(self) -> None:
        self.numeric_cols = [
            "CenterID",
            "Age",
            "Tobacco Consumption",
            "Alcohol Consumption",
            "Performance Status",
        ]
        self.categorical_cols = [
            "Gender",
            "Treatment",
            "HPV Status",
        ]
        self.active_numeric_cols: list[str] = []
        self.active_categorical_cols: list[str] = []
        self.dropped_all_missing_cols: list[str] = []
        self.outcome_cols = ["Relapse", "RFS", "T-stage", "N-stage"]
        self.id_col = "PatientID"
        self.stage_encodings = StageEncodings()
        self.transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def fit(self, dataframe: pd.DataFrame) -> "ClinicalPreprocessor":
        self.active_numeric_cols = [col for col in self.numeric_cols if not dataframe[col].isna().all()]
        self.active_categorical_cols = [col for col in self.categorical_cols if not dataframe[col].isna().all()]
        self.dropped_all_missing_cols = [
            col for col in self.numeric_cols + self.categorical_cols
            if col not in self.active_numeric_cols + self.active_categorical_cols
        ]

        numeric_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers = []
        if self.active_numeric_cols:
            transformers.append(("num", numeric_pipeline, self.active_numeric_cols))
        if self.active_categorical_cols:
            transformers.append(("cat", categorical_pipeline, self.active_categorical_cols))
        self.transformer = ColumnTransformer(transformers)
        self.transformer.fit(dataframe[self.active_numeric_cols + self.active_categorical_cols])
        self.feature_names_ = self._build_feature_names()
        return self

    def transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("ClinicalPreprocessor must be fitted before transform().")
        features = self.transformer.transform(dataframe[self.active_numeric_cols + self.active_categorical_cols])
        return np.asarray(features, dtype=np.float32)

    def fit_transform(self, dataframe: pd.DataFrame) -> np.ndarray:
        self.fit(dataframe)
        return self.transform(dataframe)

    def attach_targets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        enriched = dataframe.copy()
        enriched["t_stage_label"] = enriched["T-stage"].apply(self.stage_encodings.encode_t)
        enriched["n_stage_label"] = enriched["N-stage"].apply(self.stage_encodings.encode_n)
        relapse = enriched["Relapse"].fillna(0).astype(int)
        enriched["relapse_label"] = relapse
        enriched["rfs_time"] = enriched["RFS"].fillna(enriched["RFS"].median())
        enriched["rfs_event"] = relapse
        return enriched

    def save(self, path: str | Path) -> None:
        payload = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "active_numeric_cols": self.active_numeric_cols,
            "active_categorical_cols": self.active_categorical_cols,
            "dropped_all_missing_cols": self.dropped_all_missing_cols,
            "feature_names": self.feature_names_,
            "transformer": self.transformer,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "ClinicalPreprocessor":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls()
        instance.numeric_cols = payload["numeric_cols"]
        instance.categorical_cols = payload["categorical_cols"]
        instance.active_numeric_cols = payload.get("active_numeric_cols", list(instance.numeric_cols))
        instance.active_categorical_cols = payload.get("active_categorical_cols", list(instance.categorical_cols))
        instance.dropped_all_missing_cols = payload.get("dropped_all_missing_cols", [])
        instance.feature_names_ = payload["feature_names"]
        instance.transformer = payload["transformer"]
        return instance

    def export_summary(self, dataframe: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_rows": int(len(dataframe)),
            "missing_counts": dataframe.isna().sum().to_dict(),
            "feature_names": self.feature_names_,
            "dropped_all_missing_cols": self.dropped_all_missing_cols,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _build_feature_names(self) -> list[str]:
        if self.transformer is None:
            return []
        numeric_names = list(self.active_numeric_cols)
        categorical_names: list[str] = []
        if "cat" in self.transformer.named_transformers_:
            encoder: OneHotEncoder = self.transformer.named_transformers_["cat"].named_steps["encoder"]
            categorical_names = list(encoder.get_feature_names_out(self.active_categorical_cols))
        return numeric_names + categorical_names
