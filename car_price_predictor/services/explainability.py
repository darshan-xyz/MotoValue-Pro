"""
Feature Importance & Explainability Service for CarPricePredictor (R3).

Computes global model feature importances and local vehicle valuation attributions,
reconciling base ML model tree drivers with post-processing pricing heuristics.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping
import numpy as np
import pandas as pd

from car_price_predictor.schemas.prediction import (
    ConfidenceInterval,
    ExplainResponse,
    FeatureContribution,
    PredictionRequest,
    PricingAdjustmentsBreakdown,
)
from car_price_predictor.services.model_service import ModelService
from car_price_predictor.services.pricing import PricingService

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Decomposes vehicle valuations into constituent drivers across both the
    14-feature Random Forest ML model and domain pricing heuristic rules.
    """

    BASELINE_VEHICLE: dict[str, Any] = {
        "brand": "Maruti",
        "model": "Swift",
        "year": 2018,
        "km_driven": 60000.0,
        "mileage": 18.0,
        "engine": 1200.0,
        "max_power": 75.0,
        "seats": 5,
        "fuel": "Petrol",
        "transmission": "Manual",
        "seller_type": "Individual",
        "owner": "First Owner",
    }

    FEATURE_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
        "max_power": (
            "Higher engine power increases performance and market valuation",
            "Lower engine power reduces valuation relative to market baseline",
            "Engine power is aligned with standard market baseline",
        ),
        "km_driven": (
            "Low distance traveled preserves vehicle mechanical condition and value",
            "Higher distance driven incurs vehicle wear and reduces valuation",
            "Vehicle mileage is average for market baseline",
        ),
        "mileage": (
            "Superior fuel efficiency reduces operating costs and boosts resale value",
            "Lower fuel economy increases running costs and reduces resale value",
            "Fuel economy matches market standard baseline",
        ),
        "engine": (
            "Larger engine displacement commands premium resale pricing",
            "Smaller engine displacement commands economy market pricing",
            "Engine displacement matches standard baseline",
        ),
        "brand": (
            "Reputable manufacturer brand equity enhances residual resale value",
            "Brand positioning commands economy pricing",
            "Brand aligns with market median",
        ),
        "model": (
            "High-demand vehicle model retains stronger market value",
            "Model segment commands lower resale pricing",
            "Model pricing aligns with baseline segment",
        ),
        "transmission": (
            "Automatic transmission adds convenience and commands price premium",
            "Manual transmission positions vehicle in cost-efficient segment",
            "Transmission type aligns with standard baseline",
        ),
        "fuel": (
            "Fuel type configuration commands favorable market resale pricing",
            "Fuel type configuration incurs lower demand in current market",
            "Fuel type aligns with baseline pricing",
        ),
        "seats": (
            "Seating capacity provides expanded passenger utility",
            "Compact passenger seating capacity",
            "Standard 5-passenger seating capacity",
        ),
        "seller_type": (
            "Dealer network warranty and trust enhances transaction value",
            "Individual seller pricing reflects direct private sale",
            "Standard seller classification",
        ),
        "year": (
            "Recent model year commands higher market valuation",
            "Vehicle age causes depreciation according to standard rate",
            "Vehicle matches baseline model year (2018)",
        ),
        "owner": (
            "Single owner history commands top resale tier",
            "Multiple prior owners accelerate vehicle depreciation",
            "Standard ownership tier",
        ),
    }

    def __init__(self, model_service: ModelService) -> None:
        self.model_service = model_service
        self._cached_feature_importances: dict[str, float] | None = None
        self._cached_model_id: int | None = None

    @staticmethod
    def _to_dict(payload: Any) -> dict[str, Any]:
        """Converts inbound payload into a dictionary."""
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        if isinstance(payload, dict):
            return dict(payload)
        return dict(payload)

    def get_global_feature_importances(self) -> dict[str, float]:
        """
        Extracts Gini feature importances from the trained Random Forest model.
        Returns a dictionary mapping feature column names to importance scores.
        """
        if (
            self._cached_feature_importances is not None
            and self.model_service.is_ready()
            and id(self.model_service.model) == self._cached_model_id
        ):
            return dict(self._cached_feature_importances)

        if not self.model_service.is_ready():
            return {}

        model = self.model_service.model
        if not hasattr(model, "feature_importances_"):
            return {}

        importances = model.feature_importances_
        cols = self.model_service.feature_columns
        if len(importances) != len(cols):
            return {}

        self._cached_feature_importances = {
            str(col): round(float(val), 6) for col, val in zip(cols, importances)
        }
        self._cached_model_id = id(model)
        return dict(self._cached_feature_importances)

    def explain_prediction(
        self,
        payload: PredictionRequest | Mapping[str, Any] | dict[str, Any],
        confidence_level: float = 0.80,
    ) -> ExplainResponse:
        """
        Generates full explanation response for a vehicle payload.
        Computes local feature contributions, pricing adjustments breakdown,
        and uncertainty intervals.
        """
        data = self._to_dict(payload)
        preprocessor = self.model_service.preprocessor
        if preprocessor is None or not self.model_service.is_ready():
            raise RuntimeError("ModelService is not ready for explainability analysis")

        # 1. Base ML model inference and confidence interval
        input_df = preprocessor.preprocess(data)
        base_ml_valuation, lower_base, upper_base = self.model_service.predict_with_interval(
            input_df, confidence_level=confidence_level
        )

        # 2. Pricing heuristics
        year, mileage, owner = preprocessor.extract_pricing_inputs(data)
        owner_mult = PricingService.get_owner_multiplier(owner)
        year_factor = PricingService.get_year_factor(year)
        mileage_factor = PricingService.get_mileage_factor(mileage)
        total_mult = owner_mult * year_factor * mileage_factor

        final_valuation, lower_bound, upper_bound = PricingService.calculate_price_interval(
            base_prediction=base_ml_valuation,
            lower_base=lower_base,
            upper_base=upper_base,
            year=year,
            mileage=mileage,
            owner=owner,
        )

        formatted_price = PricingService.format_price(final_valuation)
        formatted_lower = PricingService.format_price(lower_bound)
        formatted_upper = PricingService.format_price(upper_bound)
        net_adj = round(final_valuation - base_ml_valuation, 2)

        conf_interval = ConfidenceInterval(
            lower_bound=round(lower_bound, 2),
            upper_bound=round(upper_bound, 2),
            formatted_lower=formatted_lower,
            formatted_upper=formatted_upper,
            confidence_level=confidence_level,
            method="ensemble_tree_percentiles",
        )

        pricing_adj = PricingAdjustmentsBreakdown(
            owner_multiplier=round(owner_mult, 4),
            year_factor=round(year_factor, 4),
            mileage_factor=round(mileage_factor, 4),
            total_heuristic_multiplier=round(total_mult, 6),
            net_adjustment_amount=net_adj,
        )

        # 3. Local Feature Contributions
        contributions: list[FeatureContribution] = []
        global_importances = self.get_global_feature_importances()

        # Pricing rule 1: Owner effect
        owner_impact = round(base_ml_valuation * (owner_mult - 1.0), 2)
        pct_owner = round((owner_impact / max(final_valuation, 1.0)) * 100.0, 2)
        dir_owner = "positive" if owner_impact > 10.0 else ("negative" if owner_impact < -10.0 else "neutral")
        desc_owner = (
            self.FEATURE_DESCRIPTIONS["owner"][0]
            if dir_owner == "positive"
            else (self.FEATURE_DESCRIPTIONS["owner"][1] if dir_owner == "negative" else self.FEATURE_DESCRIPTIONS["owner"][2])
        )
        contributions.append(
            FeatureContribution(
                feature="owner",
                value=owner,
                impact_amount=owner_impact,
                percentage_impact=pct_owner,
                direction=dir_owner,
                source="pricing_rule",
                description=desc_owner,
                importance=0.05,
            )
        )

        # Pricing rule 2: Year effect
        year_impact = round(base_ml_valuation * owner_mult * (year_factor - 1.0), 2)
        pct_year = round((year_impact / max(final_valuation, 1.0)) * 100.0, 2)
        dir_year = "positive" if year_impact > 10.0 else ("negative" if year_impact < -10.0 else "neutral")
        desc_year = (
            self.FEATURE_DESCRIPTIONS["year"][0]
            if dir_year == "positive"
            else (self.FEATURE_DESCRIPTIONS["year"][1] if dir_year == "negative" else self.FEATURE_DESCRIPTIONS["year"][2])
        )
        contributions.append(
            FeatureContribution(
                feature="year",
                value=year,
                impact_amount=year_impact,
                percentage_impact=pct_year,
                direction=dir_year,
                source="pricing_rule",
                description=desc_year,
                importance=0.08,
            )
        )

        # Pricing rule 3: Mileage efficiency effect (additive exact reconciliation with net_adjustment_amount)
        mileage_rule_impact = round(net_adj - (owner_impact + year_impact), 2)
        pct_mileage_rule = round((mileage_rule_impact / max(final_valuation, 1.0)) * 100.0, 2)
        dir_m_rule = "positive" if mileage_rule_impact > 10.0 else ("negative" if mileage_rule_impact < -10.0 else "neutral")
        desc_m_rule = (
            self.FEATURE_DESCRIPTIONS["mileage"][0]
            if dir_m_rule == "positive"
            else (self.FEATURE_DESCRIPTIONS["mileage"][1] if dir_m_rule == "negative" else self.FEATURE_DESCRIPTIONS["mileage"][2])
        )
        contributions.append(
            FeatureContribution(
                feature="mileage",
                value=mileage,
                impact_amount=mileage_rule_impact,
                percentage_impact=pct_mileage_rule,
                direction=dir_m_rule,
                source="pricing_rule",
                description=desc_m_rule,
                importance=0.05,
            )
        )

        # ML Features: sensitivity perturbation against baseline
        ml_features_to_evaluate = [
            "max_power",
            "km_driven",
            "mileage",
            "engine",
            "brand",
            "model",
            "fuel",
            "transmission",
            "seller_type",
            "seats",
        ]

        perturbed_payloads = []
        for feat in ml_features_to_evaluate:
            p_data = dict(data)
            p_data[feat] = self.BASELINE_VEHICLE[feat]
            perturbed_payloads.append(p_data)

        batch_df = preprocessor.preprocess_batch(perturbed_payloads)
        pert_preds = self.model_service.predict(batch_df)

        for feat, pert_base_pred in zip(ml_features_to_evaluate, pert_preds):
            feat_val = data.get(feat, self.BASELINE_VEHICLE.get(feat))
            impact_amt = (base_ml_valuation - float(pert_base_pred)) * total_mult
            pct_impact = round((impact_amt / max(final_valuation, 1.0)) * 100.0, 2)
            direction = (
                "positive"
                if impact_amt > 25.0
                else ("negative" if impact_amt < -25.0 else "neutral")
            )

            desc_tuple = self.FEATURE_DESCRIPTIONS.get(
                feat,
                (
                    f"{feat} increases valuation",
                    f"{feat} reduces valuation",
                    f"{feat} matches baseline",
                ),
            )
            description = (
                desc_tuple[0]
                if direction == "positive"
                else (desc_tuple[1] if direction == "negative" else desc_tuple[2])
            )

            feat_importance = global_importances.get(feat, None)
            if feat_importance is None and feat == "brand":
                feat_importance = global_importances.get("brand_encoded")
            elif feat_importance is None and feat == "model":
                feat_importance = global_importances.get("model_encoded")

            contributions.append(
                FeatureContribution(
                    feature=feat,
                    value=feat_val,
                    impact_amount=round(impact_amt, 2),
                    percentage_impact=pct_impact,
                    direction=direction,
                    source="ml_model",
                    description=description,
                    importance=feat_importance,
                )
            )

        # Sort contributions by absolute impact descending
        contributions.sort(key=lambda c: abs(c.impact_amount), reverse=True)

        # Identify top 3 non-neutral unique drivers
        top_drivers: list[str] = []
        for c in contributions:
            if c.feature not in top_drivers and c.direction != "neutral":
                top_drivers.append(c.feature)
            if len(top_drivers) == 3:
                break

        return ExplainResponse(
            price=formatted_price,
            valuation=round(final_valuation, 2),
            currency="INR",
            base_ml_valuation=round(base_ml_valuation, 2),
            base_ml_price=round(base_ml_valuation, 2),
            confidence_interval=conf_interval,
            feature_contributions=contributions,
            top_drivers=top_drivers,
            pricing_adjustments=pricing_adj,
            global_feature_importances=global_importances,
            status="success",
        )
