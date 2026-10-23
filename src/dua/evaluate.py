"""Evaluation harness for the Document Understanding Assistant.

Runs inference over a dataset split, computes structured metrics, and
writes a dated evidence report (.md + .json) to the output directory.
No API calls or data reads live in this module; they are injected via
the ``Runner`` and ``Dataset`` abstractions so the file itself stays
pure and testable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dua import schemas
from dua.baseline import extract as baseline_extract
from dua.guardrails import check_arithmetic, check_missing_core_fields


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single ground-truth example consumed by the evaluator."""

    doc_id: str
    doc_type: str
    fields: Dict[str, Any]
    body_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A model prediction for one example."""

    doc_id: str
    doc_type: str
    fields: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldMetric:
    """Per-field precision/recall/F1 plus support counts."""

    field_name: str
    precision: float
    recall: float
    f1: float
    support: int
    correct: int


@dataclass
class ConfusionEntry:
    """One cell of the doc-type confusion matrix."""

    expected: str
    predicted: str
    count: int


@dataclass
class ReviewTriggerBreakdown:
    """Counts of review triggers grouped by reason."""

    total: int = 0
    low_confidence: int = 0
    schema_violation: int = 0
    missing_field: int = 0
    other: int = 0


@dataclass
class EvaluationResult:
    """Aggregated metrics for one evaluation run."""

    split: str
    model: str
    limit: Optional[int]
    total_examples: int
    evaluated_examples: int
    field_metrics: List[FieldMetric]
    doc_type_accuracy: float
    confusion_matrix: List[ConfusionEntry]
    schema_valid_rate: float
    review_rate: float
    review_breakdown: ReviewTriggerBreakdown
    value_present_accuracy: float
    arithmetic_consistency_rate: float
    failure_examples: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Abstractions (injectable, swappable)
# ---------------------------------------------------------------------------

class Dataset:
    """Loads ground-truth examples for a given split.

    Subclasses or callables passed to ``Evaluator`` implement the actual
    file/network access; this base class only documents the contract.
    """

    def __init__(self, split: str) -> None:
        self.split = split

    def load(self, limit: Optional[int] = None) -> Iterable[Example]:
        """Yield up to ``limit`` examples for this split."""
        raise NotImplementedError


class Runner:
    """Runs a model prediction for a single example.

    Implementations wrap the actual LLM call (e.g. OpenRouter via
    ``OPENAI_API_KEY``). The evaluator never touches the network itself.
    """

    def __init__(self, model: str) -> None:
        self.model = model

    def predict(self, example: Example) -> Prediction:
        """Return a ``Prediction`` for ``example``."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def exact_match(expected: Any, predicted: Any) -> bool:
    """Return True when two values are considered equal for scoring."""
    return expected == predicted


def _is_present(value: Any) -> bool:
    """True when a field value is considered present (non-empty)."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    return True


def _pair_by_id(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> List[Tuple[Example, Prediction]]:
    """Match examples to predictions by ``doc_id``, preserving example order."""
    by_id = {p.doc_id: p for p in predictions}
    pairs: List[Tuple[Example, Prediction]] = []
    for ex in examples:
        pred = by_id.get(ex.doc_id)
        if pred is None:
            pred = Prediction(doc_id=ex.doc_id, doc_type="", fields={})
        pairs.append((ex, pred))
    return pairs


def compute_field_metrics(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> List[FieldMetric]:
    """Compute precision/recall/F1 for every field seen across the data.

    A field is counted as present when its value is non-empty. A true
    positive requires both sides to be present and ``exact_match`` to hold.
    Returns an empty list for empty input.
    """
    pairs = _pair_by_id(examples, predictions)
    if not pairs:
        return []

    field_names: set = set()
    for ex, _ in pairs:
        field_names.update(ex.fields.keys())
    for _, pred in pairs:
        field_names.update(pred.fields.keys())

    metrics: List[FieldMetric] = []
    for name in sorted(field_names):
        tp = fp = fn = 0
        for ex, pred in pairs:
            ev = ex.fields.get(name)
            pv = pred.fields.get(name)
            ev_present = _is_present(ev)
            pv_present = _is_present(pv)
            matched = ev_present and pv_present and exact_match(ev, pv)
            if matched:
                tp += 1
            elif pv_present:
                fp += 1
            if ev_present and not matched:
                fn += 1
        support = sum(1 for ex, _ in pairs if _is_present(ex.fields.get(name)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics.append(
            FieldMetric(
                field_name=name,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=support,
                correct=tp,
            )
        )
    return metrics


def compute_doc_type_accuracy(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> Tuple[float, List[ConfusionEntry]]:
    """Return overall doc-type accuracy and the confusion matrix rows.

    Returns ``(0.0, [])`` for empty input.
    """
    pairs = _pair_by_id(examples, predictions)
    if not pairs:
        return 0.0, []

    correct = 0
    cells: Dict[Tuple[str, str], int] = {}
    for ex, pred in pairs:
        expected = ex.doc_type
        predicted = pred.doc_type
        if expected == predicted:
            correct += 1
        cells[(expected, predicted)] = cells.get((expected, predicted), 0) + 1

    accuracy = correct / len(pairs)
    confusion = [
        ConfusionEntry(expected=e, predicted=p, count=c)
        for (e, p), c in sorted(cells.items())
    ]
    return round(accuracy, 4), confusion


def compute_schema_valid_rate(predictions: Sequence[Prediction]) -> float:
    """Fraction of predictions that pass schema validation.

    A prediction is schema-valid when its ``doc_type`` maps to one of the
    allowed doc types. Returns ``0.0`` for empty input.
    """
    if not predictions:
        return 0.0
    valid = 0
    for pred in predictions:
        dt = schemas.normalize_doc_type(pred.doc_type)
        if dt in schemas.ALLOWED_DOC_TYPES:
            valid += 1
    return round(valid / len(predictions), 4)


def compute_review_rate_and_breakdown(
    predictions: Sequence[Prediction],
) -> Tuple[float, ReviewTriggerBreakdown]:
    """Review rate plus the count of each trigger reason.

    Review triggers are read from ``prediction.raw`` (keys: ``confidence``,
    ``needs_human_review``, ``issues``). Returns ``(0.0, ReviewTriggerBreakdown())``
    for empty input.
    """
    breakdown = ReviewTriggerBreakdown()
    if not predictions:
        return 0.0, breakdown

    reviewed = 0
    for pred in predictions:
        raw = pred.raw or {}
        confidence = raw.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        needs_review = bool(raw.get("needs_human_review"))
        issues = list(raw.get("issues") or [])

        triggered = False
        reasons: List[str] = []

        if confidence is not None and confidence < 0.6:
            triggered = True
            reasons.append("low_confidence")
            breakdown.low_confidence += 1

        dt = schemas.normalize_doc_type(pred.doc_type)
        if dt is None or dt not in schemas.ALLOWED_DOC_TYPES:
            triggered = True
            reasons.append("schema_violation")
            breakdown.schema_violation += 1

        missing = False
        for issue in issues:
            if isinstance(issue, str) and "missing_required_field" in issue:
                missing = True
        if not missing:
            _, miss_issues = check_missing_core_fields(pred.doc_type, pred.fields)
            if miss_issues:
                missing = True
        if missing:
            triggered = True
            reasons.append("missing_field")
            breakdown.missing_field += 1

        other_issues = [
            i for i in issues
            if not (isinstance(i, str) and "missing_required_field" in i)
        ]
        if other_issues:
            triggered = True
            reasons.append("other")
            breakdown.other += 1

        if needs_review or triggered:
            reviewed += 1
        breakdown.total += 1

    rate = reviewed / len(predictions) if predictions else 0.0
    return round(rate, 4), breakdown


def compute_value_present_accuracy(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> float:
    """Accuracy restricted to fields whose value actually appears in the body.

    For each example, consider only fields whose string value is a substring
    of ``body_text`` (case-insensitive). A prediction is correct for such a
    field when it exactly matches the ground truth. Returns ``0.0`` when no
    body-present fields exist or input is empty.
    """
    pairs = _pair_by_id(examples, predictions)
    if not pairs:
        return 0.0

    total = correct = 0
    for ex, pred in pairs:
        body = (ex.body_text or "").lower()
        for name, ev in ex.fields.items():
            if not _is_present(ev):
                continue
            if str(ev).lower() not in body:
                continue
            total += 1
            pv = pred.fields.get(name)
            if _is_present(pv) and exact_match(ev, pv):
                correct += 1
    return round(correct / total, 4) if total else 0.0


def compute_arithmetic_consistency_rate(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
) -> float:
    """Fraction of numeric-derived fields that reconcile with the body text.

    Uses :func:`dua.guardrails.check_arithmetic` on each prediction's fields.
    Only predictions carrying subtotal/tax/total are counted. Returns ``0.0``
    for empty input or when no numeric-derived fields are present.
    """
    if not predictions:
        return 0.0

    total = consistent = 0
    for pred in predictions:
        fields = pred.fields or {}
        has_numeric = any(
            _is_present(fields.get(k)) for k in ("subtotal", "tax", "total")
        )
        if not has_numeric:
            continue
        total += 1
        ok, _ = check_arithmetic(fields)
        if ok:
            consistent += 1
    return round(consistent / total, 4) if total else 0.0


def rank_failures(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top failure examples, most severe first.

    Severity scoring: doc-type mismatch (+3), missing core field (+2),
    per-field mismatch (+1). Returns an empty list for empty input.
    """
    pairs = _pair_by_id(examples, predictions)
    if not pairs:
        return []

    failures: List[Dict[str, Any]] = []
    for ex, pred in pairs:
        severity = 0
        field_errors: List[Dict[str, Any]] = []
        if ex.doc_type != pred.doc_type:
            severity += 3
        for name in sorted(set(ex.fields) | set(pred.fields)):
            ev = ex.fields.get(name)
            pv = pred.fields.get(name)
            if _is_present(ev) and not (_is_present(pv) and exact_match(ev, pv)):
                severity += 1
                field_errors.append(
                    {
                        "field": name,
                        "expected": ev,
                        "predicted": pv,
                    }
                )
        _, miss_issues = check_missing_core_fields(pred.doc_type, pred.fields)
        if miss_issues:
            severity += 2

        if severity == 0:
            continue

        failures.append(
            {
                "doc_id": ex.doc_id,
                "expected_doc_type": ex.doc_type,
                "predicted_doc_type": pred.doc_type,
                "severity": severity,
                "field_errors": field_errors,
                "reason": (
                    "doc_type_mismatch" if ex.doc_type != pred.doc_type else "field_mismatch"
                ),
            }
        )

    failures.sort(key=lambda f: f["severity"], reverse=True)
    return failures[:limit]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Orchestrates a full evaluation run against a dataset split."""

    def __init__(
        self,
        dataset: Dataset,
        runner: Runner,
        output_dir: Path,
    ) -> None:
        self.dataset = dataset
        self.runner = runner
        self.output_dir = Path(output_dir)

    def run(self, limit: Optional[int] = None) -> EvaluationResult:
        """Run inference and compute all metrics."""
        examples = list(self.dataset.load(limit=limit))
        predictions = [self.runner.predict(ex) for ex in examples]
        pairs = _pair_by_id(examples, predictions)

        field_metrics = compute_field_metrics(examples, predictions)
        doc_type_accuracy, confusion_matrix = compute_doc_type_accuracy(examples, predictions)
        schema_valid_rate = compute_schema_valid_rate(predictions)
        review_rate, review_breakdown = compute_review_rate_and_breakdown(predictions)
        value_present_accuracy = compute_value_present_accuracy(examples, predictions)
        arithmetic_consistency_rate = compute_arithmetic_consistency_rate(examples, predictions)
        failure_examples = rank_failures(examples, predictions)

        return EvaluationResult(
            split=self.dataset.split,
            model=self.runner.model,
            limit=limit,
            total_examples=len(examples),
            evaluated_examples=len(predictions),
            field_metrics=field_metrics,
            doc_type_accuracy=doc_type_accuracy,
            confusion_matrix=confusion_matrix,
            schema_valid_rate=schema_valid_rate,
            review_rate=review_rate,
            review_breakdown=review_breakdown,
            value_present_accuracy=value_present_accuracy,
            arithmetic_consistency_rate=arithmetic_consistency_rate,
            failure_examples=failure_examples,
        )

    def render_markdown(self, result: EvaluationResult) -> str:
        """Render the human-readable evidence report."""
        lines: List[str] = []
        lines.append(f"# Evaluation Report: {result.split}")
        lines.append("")
        lines.append(f"- **Model:** {result.model}")
        lines.append(f"- **Limit:** {result.limit if result.limit is not None else 'none'}")
        lines.append(f"- **Total examples:** {result.total_examples}")
        lines.append(f"- **Evaluated examples:** {result.evaluated_examples}")
        lines.append("")
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append(f"- **Doc-type accuracy:** {result.doc_type_accuracy}")
        lines.append(f"- **Schema valid rate:** {result.schema_valid_rate}")
        lines.append(f"- **Review rate:** {result.review_rate}")
        lines.append(f"- **Value-present accuracy:** {result.value_present_accuracy}")
        lines.append(f"- **Arithmetic consistency rate:** {result.arithmetic_consistency_rate}")
        lines.append("")
        lines.append("### Review Trigger Breakdown")
        lines.append("")
        rb = result.review_breakdown
        lines.append(f"- **Total reviewed:** {rb.total}")
        lines.append(f"- **Low confidence:** {rb.low_confidence}")
        lines.append(f"- **Schema violation:** {rb.schema_violation}")
        lines.append(f"- **Missing field:** {rb.missing_field}")
        lines.append(f"- **Other:** {rb.other}")
        lines.append("")
        lines.append("## Field Metrics")
        lines.append("")
        lines.append("| Field | Precision | Recall | F1 | Support | Correct |")
        lines.append("|-------|-----------|--------|----|---------|---------|")
        for fm in result.field_metrics:
            lines.append(
                f"| {fm.field_name} | {fm.precision} | {fm.recall} | {fm.f1} | "
                f"{fm.support} | {fm.correct} |"
            )
        lines.append("")
        lines.append("## Confusion Matrix (Doc Type)")
        lines.append("")
        lines.append("| Expected | Predicted | Count |")
        lines.append("|----------|-----------|-------|")
        for entry in result.confusion_matrix:
            lines.append(f"| {entry.expected} | {entry.predicted} | {entry.count} |")
        lines.append("")
        lines.append("## Top Failures")
        lines.append("")
        if result.failure_examples:
            lines.append("| Doc ID | Expected | Predicted | Severity | Reason |")
            lines.append("|--------|----------|-----------|----------|--------|")
            for fe in result.failure_examples:
                lines.append(
                    f"| {fe['doc_id']} | {fe['expected_doc_type']} | "
                    f"{fe['predicted_doc_type']} | {fe['severity']} | {fe['reason']} |"
                )
        else:
            lines.append("_No failures._")
        lines.append("")
        return "\n".join(lines)

    def render_json(self, result: EvaluationResult) -> str:
        """Render the machine-readable evidence dump."""
        return json.dumps(asdict(result), indent=2, default=str)

    def save(self, result: EvaluationResult) -> Tuple[Path, Path]:
        """Write dated .md and .json reports into ``output_dir``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        base = f"eval_{result.split}_{stamp}"
        md_path = self.output_dir / f"{base}.md"
        json_path = self.output_dir / f"{base}.json"
        md_path.write_text(self.render_markdown(result), encoding="utf-8")
        json_path.write_text(self.render_json(result), encoding="utf-8")
        return md_path, json_path


# ---------------------------------------------------------------------------
# Concrete Dataset / Runner implementations used by the CLI
# ---------------------------------------------------------------------------

class LabelDataset(Dataset):
    """Load ground-truth examples from ``data/<split>/labels/*.json``."""

    def __init__(self, split: str, data_root: Optional[Path] = None) -> None:
        super().__init__(split)
        self.data_root = Path(data_root) if data_root else Path("data")

    def load(self, limit: Optional[int] = None) -> Iterable[Example]:
        labels_dir = self.data_root / self.split / "labels"
        paths = sorted(labels_dir.glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = data.get("fields", {}) or {}
            body_text = data.get("body_text", "")
            if not body_text:
                # best-effort: read the document file referenced by the label
                doc_file = self.data_root / self.split / "docs" / data.get("file", "")
                if doc_file.exists():
                    from dua.baseline import read_document

                    body_text = read_document(doc_file)
            yield Example(
                doc_id=data.get("doc_id", path.stem),
                doc_type=data.get("doc_type", ""),
                fields=fields,
                body_text=body_text,
                metadata=data.get("meta", {}),
            )


class BaselineRunner(Runner):
    """Rule-based runner using :func:`dua.baseline.extract` (no API calls).

    The baseline is deterministic and carries no confidence score, so
    ``raw`` sets ``confidence`` to ``None``. The evaluator excludes
    ``None`` confidence from low-confidence review triggers.
    """

    def predict(self, example: Example) -> Prediction:
        out = baseline_extract(example.body_text)
        fields = out.get("fields", {}) or {}
        raw = dict(out)
        raw["confidence"] = None
        return Prediction(
            doc_id=example.doc_id,
            doc_type=out.get("doc_type", ""),
            fields=fields,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluate CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Document Understanding Assistant on a data split.",
    )
    parser.add_argument(
        "--split",
        default="dev",
        help="data split to evaluate (dev, test, challenge)",
    )
    parser.add_argument(
        "--model",
        default="baseline",
        help="model identifier recorded in the report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="evidence",
        help="directory for dated .md and .json reports",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="root directory containing data/<split>/labels",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: parse args, run evaluation, save evidence."""
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = LabelDataset(split=args.split, data_root=Path(args.data_root))
    runner = BaselineRunner(model=args.model)

    evaluator = Evaluator(dataset=dataset, runner=runner, output_dir=Path(args.output_dir))
    result = evaluator.run(limit=args.limit)

    md_path, json_path = evaluator.save(result)
    print(f"Evaluation complete: {result.split} split, {result.evaluated_examples} examples")
    print(f"  doc-type accuracy: {result.doc_type_accuracy}")
    print(f"  schema valid rate: {result.schema_valid_rate}")
    print(f"  review rate: {result.review_rate}")
    print(f"  reports: {md_path}, {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
