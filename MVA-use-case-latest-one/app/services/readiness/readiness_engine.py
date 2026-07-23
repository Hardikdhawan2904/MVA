"""AI Readiness engine — computes analytics, ML, LLM, and overall readiness."""

from typing import Any

from app.core.constants import READINESS_READY_THRESHOLD, READINESS_PARTIALLY_READY_THRESHOLD
from app.core.enums import ReadinessType, ReadinessStatus
from app.services.profiling.column_profiler import ColumnProfileResult


class ReadinessResult:
    """Result of a single readiness assessment."""

    def __init__(
        self,
        assessment_type: ReadinessType,
        score: float,
        status: ReadinessStatus,
        strengths: list[dict[str, Any]],
        blocking_issues: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        weight_profile_version: str,
        dataset_score: float | None = None,
        task_compatibility_score: float | None = None,
    ):
        self.assessment_type = assessment_type
        self.score = score
        self.status = status
        self.strengths = strengths
        self.blocking_issues = blocking_issues
        self.recommendations = recommendations
        self.evidence = evidence
        self.weight_profile_version = weight_profile_version
        # dataset_score: the question-independent component, 0-100, scaled
        # against only the dataset-only points actually assessed (so a
        # skipped dimension doesn't silently drag this down). None only
        # when literally nothing dataset-only was assessable.
        # task_compatibility_score: the question-dependent component, 0-100.
        # None when no business_question/feature_recommendation was ever in
        # play -- there's no task to be compatible with, not a score of 0.
        # `score` itself is unchanged by this split -- same formula, same
        # value as before; these two are purely additive transparency.
        self.dataset_score = dataset_score
        self.task_compatibility_score = task_compatibility_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_type": self.assessment_type.value,
            "score": round(self.score, 2),
            "status": self.status.value,
            "strengths": self.strengths,
            "blocking_issues": self.blocking_issues,
            "recommendations": self.recommendations,
            "evidence": self.evidence,
            "weight_profile_version": self.weight_profile_version,
            "dataset_score": round(self.dataset_score, 2) if self.dataset_score is not None else None,
            "task_compatibility_score": (
                round(self.task_compatibility_score, 2) if self.task_compatibility_score is not None else None
            ),
        }


def _determine_status(score: float) -> ReadinessStatus:
    """Map score to readiness status using configured thresholds."""
    if score >= READINESS_READY_THRESHOLD:
        return ReadinessStatus.READY
    if score >= READINESS_PARTIALLY_READY_THRESHOLD:
        return ReadinessStatus.PARTIALLY_READY
    return ReadinessStatus.NOT_READY


class ReadinessEngine:
    """Computes all readiness assessments from shared profile/quality evidence."""

    def assess_all(
        self,
        profiles: list[ColumnProfileResult],
        quality_results: list[dict[str, Any]],
        grain_columns: list[str],
        has_temporal: bool,
        metric_count: int,
        dimension_count: int,
        description_coverage: float,
        row_count: int,
        feature_recommendation: dict[str, Any] | None = None,
    ) -> list[ReadinessResult]:
        """Compute analytics, ML, LLM, and overall readiness.

        feature_recommendation is the feature_target_agent's target/feature/drop
        classification (see app/agents/data_profiling_agent/nodes/pipeline.py's
        recommend_target_features node) — always populated (deterministically when
        no business question was asked, LLM-derived otherwise), so ml/llm readiness
        can reflect an actual analysis plan instead of a blind heuristic. None only
        for callers that haven't been updated to pass it (e.g. legacy tests)."""
        analytics = self._assess_analytics(
            profiles, quality_results, has_temporal, metric_count, dimension_count, grain_columns
        )
        ml = self._assess_ml(profiles, quality_results, grain_columns, row_count, feature_recommendation)
        llm = self._assess_llm(profiles, quality_results, description_coverage, feature_recommendation)

        overall_score = (analytics.score + ml.score + llm.score) / 3.0
        overall = ReadinessResult(
            assessment_type=ReadinessType.OVERALL,
            score=overall_score,
            status=_determine_status(overall_score),
            strengths=[],
            blocking_issues=[],
            recommendations=[],
            evidence=[
                {"component": "analytics", "score": round(analytics.score, 2)},
                {"component": "ml", "score": round(ml.score, 2)},
                {"component": "llm", "score": round(llm.score, 2)},
            ],
            weight_profile_version="overall-v1",
        )

        return [analytics, ml, llm, overall]

    def _assess_analytics(
        self, profiles, quality_results, has_temporal, metric_count, dimension_count, grain_columns
    ) -> ReadinessResult:
        """Analytics readiness emphasizes completeness, dimensions, metrics, grain."""
        score = 0.0
        strengths: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        # Completeness contribution
        comp = self._get_quality_score(quality_results, "completeness")
        if comp is not None:
            score += comp * 20
            if comp >= 0.9:
                strengths.append({"code": "HIGH_COMPLETENESS", "dimension": "completeness", "value": comp})
            evidence.append({"dimension": "completeness", "value": comp})

        # Validity
        val = self._get_quality_score(quality_results, "validity")
        if val is not None:
            score += val * 15
            evidence.append({"dimension": "validity", "value": val})

        # Consistency
        con = self._get_quality_score(quality_results, "consistency")
        if con is not None:
            score += con * 15
            evidence.append({"dimension": "consistency", "value": con})

        # Metrics available
        if metric_count >= 2:
            score += 15
            strengths.append({"code": "USABLE_METRICS", "value": metric_count})
        elif metric_count >= 1:
            score += 8

        # Dimensions available
        if dimension_count >= 3:
            score += 15
            strengths.append({"code": "USABLE_DIMENSIONS", "value": dimension_count})
        elif dimension_count >= 1:
            score += 8

        # Grain identified
        if grain_columns:
            score += 10
            strengths.append({"code": "IDENTIFIABLE_GRAIN", "value": grain_columns})
        else:
            blockers.append({"code": "NO_GRAIN_IDENTIFIED"})

        # Temporal fields
        if has_temporal:
            score += 10
            strengths.append({"code": "TEMPORAL_FIELDS_AVAILABLE"})
        else:
            recommendations.append({"code": "ADD_TEMPORAL_FIELD", "priority": "medium"})

        score = min(100.0, score)
        return ReadinessResult(
            assessment_type=ReadinessType.ANALYTICS,
            score=score,
            status=_determine_status(score),
            strengths=strengths,
            blocking_issues=blockers,
            recommendations=recommendations,
            evidence=evidence,
            weight_profile_version="analytics-v1",
            # Entirely dataset-driven already -- no business_question input
            # anywhere in this method -- so dataset_score is just score,
            # and there's no task component to report.
            dataset_score=score,
            task_compatibility_score=None,
        )

    def _assess_ml(self, profiles, quality_results, grain_columns, row_count, feature_recommendation=None) -> ReadinessResult:
        """ML readiness emphasizes completeness, feature coverage, identifier contamination.

        Tracked as two point pools that sum to the same overall `score` this
        method has always produced (unchanged formula/value) — dataset_pts
        for structural, question-independent facts about the file itself,
        task_pts for anything that depends on feature_recommendation (which
        only exists in its real form when a business_question was asked).
        Each pool also tracks its own max-possible points so dataset_score/
        task_compatibility_score can be reported on a fair 0-100 scale
        regardless of which dimensions happened to be assessable."""
        score = 0.0
        dataset_pts = 0.0
        dataset_max = 0.0
        task_pts = 0.0
        task_max = 0.0
        strengths: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        # Completeness — dataset-only
        comp = self._get_quality_score(quality_results, "completeness")
        if comp is not None:
            pts = comp * 20
            score += pts
            dataset_pts += pts
            dataset_max += 20
            if comp >= 0.9:
                strengths.append({"code": "HIGH_COMPLETENESS", "dimension": "completeness", "value": comp})
            evidence.append({"dimension": "completeness", "value": comp})

        # Consistency — dataset-only
        con = self._get_quality_score(quality_results, "consistency")
        if con is not None:
            pts = con * 15
            score += pts
            dataset_pts += pts
            dataset_max += 15
            evidence.append({"dimension": "consistency", "value": con})

        # Feature coverage — prefer the actual target/feature/drop classification
        # (weighted by usefulness) over the blind "every non-identifier column
        # counts" heuristic, when one is available. The former is inherently
        # task-dependent (it's literally the question's own feature plan);
        # the latter is a dataset-only structural fallback.
        recommended_features = (feature_recommendation or {}).get("feature_columns")
        has_task = recommended_features is not None
        if has_task:
            usefulness_weight = {"high": 1.0, "medium": 0.6, "low": 0.25}
            weighted = sum(usefulness_weight.get(f.get("usefulness"), 0.5) for f in recommended_features)
            feature_ratio = weighted / max(len(profiles), 1)
            pts = min(15.0, feature_ratio * 15)
            score += pts
            task_pts += pts
            task_max += 15
            evidence.append({"dimension": "feature_coverage", "value": round(feature_ratio, 3), "source": "feature_recommendation"})

            target_column = feature_recommendation.get("target_column")
            if target_column:
                strengths.append({"code": "TARGET_IDENTIFIED", "value": target_column})
            elif feature_recommendation.get("approach_reasoning"):
                # approach_reasoning is only ever set on the LLM path — its
                # presence without a target_column means a business question
                # was asked but the agent couldn't resolve a target, unlike
                # the deterministic (question-less) fallback which never sets it.
                blockers.append({"code": "NO_TARGET_IDENTIFIED"})

            recommended_drops = set(d.get("column") for d in feature_recommendation.get("drop_columns") or [])
        else:
            non_id_cols = [p for p in profiles if p.physical_name not in grain_columns]
            feature_ratio = len(non_id_cols) / max(len(profiles), 1)
            pts = feature_ratio * 15
            score += pts
            dataset_pts += pts
            dataset_max += 15
            evidence.append({"dimension": "feature_coverage", "value": round(feature_ratio, 3)})
            recommended_drops = set(grain_columns)

        # Identifier contamination check — attributed to whichever pool
        # recommended_drops came from (feature_recommendation's own
        # drop_columns when a task is in play, else the dataset-only grain).
        id_contamination = len(recommended_drops) / max(len(profiles), 1)
        penalty = 0.0
        if id_contamination > 0.3:
            blockers.append({"code": "IDENTIFIER_CONTAMINATION", "value": round(id_contamination, 3)})
            penalty = 10.0
        elif id_contamination > 0:
            recommendations.append({"code": "EXCLUDE_IDENTIFIER_FEATURES", "priority": "high"})
        score -= penalty
        if has_task:
            task_pts -= penalty
        else:
            dataset_pts -= penalty

        # Row count adequacy — dataset-only
        if row_count >= 10000:
            pts = 15.0
            strengths.append({"code": "ADEQUATE_ROW_COUNT", "value": row_count})
        elif row_count >= 1000:
            pts = 10.0
        elif row_count >= 100:
            pts = 5.0
        else:
            pts = 0.0
            blockers.append({"code": "INSUFFICIENT_ROWS", "value": row_count})
        score += pts
        dataset_pts += pts
        dataset_max += 15

        # Cardinality health — dataset-only
        high_card = sum(1 for p in profiles if p.cardinality_ratio > 0.9 and p.physical_name not in grain_columns)
        if high_card == 0:
            pts = 10.0
        else:
            recommendations.append({"code": "HIGH_CARDINALITY_FEATURES", "value": high_card, "priority": "medium"})
            pts = 5.0
        score += pts
        dataset_pts += pts
        dataset_max += 10

        # Uniqueness — dataset-only
        uniq = self._get_quality_score(quality_results, "uniqueness")
        if uniq is not None:
            pts = uniq * 10
            score += pts
            dataset_pts += pts
            dataset_max += 10
            evidence.append({"dimension": "uniqueness", "value": uniq})

        score = max(0.0, min(100.0, score))
        dataset_score = max(0.0, min(100.0, dataset_pts / dataset_max * 100)) if dataset_max > 0 else None
        task_compatibility_score = max(0.0, min(100.0, task_pts / task_max * 100)) if has_task and task_max > 0 else None
        return ReadinessResult(
            assessment_type=ReadinessType.ML,
            score=score,
            status=_determine_status(score),
            strengths=strengths,
            blocking_issues=blockers,
            recommendations=recommendations,
            evidence=evidence,
            weight_profile_version="ml-v1",
            dataset_score=dataset_score,
            task_compatibility_score=task_compatibility_score,
        )

    def _assess_llm(self, profiles, quality_results, description_coverage, feature_recommendation=None) -> ReadinessResult:
        """LLM readiness emphasizes description coverage, semantic quality, schema clarity."""
        score = 0.0
        strengths: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        recommendations: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []

        # Description coverage
        if description_coverage >= 0.8:
            score += 25
            strengths.append({"code": "HIGH_DESCRIPTION_COVERAGE", "value": round(description_coverage, 3)})
        elif description_coverage >= 0.5:
            score += 15
        else:
            blockers.append({"code": "LOW_DESCRIPTION_COVERAGE", "value": round(description_coverage, 3)})
            recommendations.append({"code": "ADD_COLUMN_DESCRIPTIONS", "priority": "high"})
            score += description_coverage * 25
        evidence.append({"dimension": "description_coverage", "value": round(description_coverage, 3)})

        # Semantic quality
        sem = self._get_quality_score(quality_results, "semantic_quality")
        if sem is not None:
            score += sem * 20
            evidence.append({"dimension": "semantic_quality", "value": sem})

        # Schema clarity (low null ratio across columns)
        avg_null = sum(p.null_ratio for p in profiles) / max(len(profiles), 1)
        schema_clarity = 1.0 - avg_null
        score += schema_clarity * 15
        evidence.append({"dimension": "schema_clarity", "value": round(schema_clarity, 3)})

        # Consistent terminology (low distinct naming patterns)
        score += 15  # Baseline — full analysis would check naming conventions

        # Safe sample availability
        samples_available = sum(1 for p in profiles if len(p.sample_values) > 0)
        sample_ratio = samples_available / max(len(profiles), 1)
        score += sample_ratio * 10
        evidence.append({"dimension": "sample_availability", "value": round(sample_ratio, 3)})

        # Context-rich metadata
        score += min(15, description_coverage * 15)

        score = max(0.0, min(100.0, score))
        # Everything above is dataset-only (description_coverage,
        # semantic_quality, schema_clarity, consistent_terminology,
        # sample_availability, context-rich metadata) and already sums to
        # a 100-point scale on its own, so it doubles directly as
        # dataset_score — no separate accumulator needed like _assess_ml's
        # variable-max dimensions.
        dataset_score = score

        # Question-suitability boost — additive only, and only when the LLM
        # sub-agent actually ran (recommended_approach == "llm"); never applies
        # to (and never lowers the score for) question-less runs. This is the
        # only task-dependent component, so task_compatibility_score is just
        # the underlying confidence rescaled to 0-100 -- the boost itself is
        # capped at 10 points, but the *compatibility* is 0-100 like every
        # other score here.
        task_compatibility_score = None
        if feature_recommendation and feature_recommendation.get("recommended_approach") == "llm":
            confidence = feature_recommendation.get("confidence") or 0.0
            boost = min(10.0, confidence * 10)
            score = min(100.0, score + boost)
            task_compatibility_score = max(0.0, min(100.0, confidence * 100))
            strengths.append({"code": "QUESTION_SUITABLE_FOR_LLM", "value": round(confidence, 3)})
            evidence.append({"dimension": "recommended_approach", "value": "llm", "confidence": round(confidence, 3)})

        return ReadinessResult(
            assessment_type=ReadinessType.LLM,
            score=score,
            status=_determine_status(score),
            strengths=strengths,
            blocking_issues=blockers,
            recommendations=recommendations,
            evidence=evidence,
            weight_profile_version="llm-v1",
            dataset_score=dataset_score,
            task_compatibility_score=task_compatibility_score,
        )

    def _get_quality_score(self, quality_results: list[dict[str, Any]], dimension: str) -> float | None:
        """Extract score for a quality dimension."""
        for r in quality_results:
            if r.get("dimension") == dimension and r.get("status") == "assessed":
                return r.get("score")
        return None
