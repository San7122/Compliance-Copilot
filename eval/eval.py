"""Runs the eval question set against a running Compliance Copilot API and reports:
  - retrieval hit rate:     for answerable questions, did the expected document show up
                             among the citations?
  - citation correctness:   for every citation returned, does its excerpt actually appear
                             (near-verbatim) in the source document on disk? (catches
                             hallucinated quotes)
  - refusal accuracy:       did answerable/not-answerable match expectation, for BOTH
                             answerable and deliberately-unanswerable questions?
  - answer correctness:     for answerable questions, does the answer text actually contain
                             the expected fact (`expect_keywords`)? A citation can point at
                             the right section while the answer still states the wrong
                             number, so this is checked independently of retrieval.

Exits non-zero if any question errored or any metric falls below --fail-under, so this
can be used directly as a CI gate on changes to prompts, chunking, or the embedding model.

Usage:
    python eval.py [--api-url http://localhost:8000] [--questions questions.yaml]
                   [--fail-under 0.8]
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

import requests
import yaml

DOCS_DIR = Path(__file__).parent.parent / "docs"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_doc_texts() -> dict[str, str]:
    """Map document title (first H1) -> normalized full text, for citation checking."""
    texts = {}
    for path in DOCS_DIR.glob("*.md"):
        raw = path.read_text(encoding="utf-8")
        match = re.search(r"^#\s+(.*)", raw, re.MULTILINE)
        title = match.group(1).strip() if match else path.stem
        texts[title] = normalize(raw)
    return texts


def excerpt_appears_in_doc(excerpt: str, doc_text: str, threshold: float = 0.7) -> bool:
    """Fuzzy substring check: does something close to `excerpt` appear in `doc_text`?
    Uses a sliding window + SequenceMatcher ratio since the LLM may paraphrase whitespace
    or trim slightly rather than copying byte-for-byte.
    """
    norm_excerpt = normalize(excerpt)
    if not norm_excerpt:
        return False
    if norm_excerpt in doc_text:
        return True

    window = len(norm_excerpt)
    best = 0.0
    step = max(1, window // 4)
    for start in range(0, max(1, len(doc_text) - window), step):
        chunk = doc_text[start : start + window + step]
        ratio = difflib.SequenceMatcher(None, norm_excerpt, chunk).ratio()
        best = max(best, ratio)
        if best >= threshold:
            return True
    return best >= threshold


def answer_matches_keywords(answer: str, keywords: list[str]) -> bool:
    """Does the answer text contain the expected fact?

    Keywords are treated as ALTERNATIVES (any match passes), because a question often has
    several equally-correct phrasings of the same fact -- e.g. ["MFA", "multi-factor"] is
    one fact written two ways, not two separate requirements.
    """
    norm_answer = normalize(answer)
    return any(normalize(kw) in norm_answer for kw in keywords)


def run(api_url: str, questions_path: Path):
    questions = yaml.safe_load(questions_path.read_text())
    doc_texts = load_doc_texts()

    rows = []
    for q in questions:
        try:
            resp = requests.post(f"{api_url}/query", json={"question": q["question"]}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "error": str(e),
                    "refusal_ok": False,
                    "retrieval_ok": None,
                    "citation_ok": None,
                    "answer_ok": None,
                }
            )
            continue

        refusal_ok = data["answerable"] == q["answerable"]

        retrieval_ok = None
        if q["answerable"]:
            cited_docs = {c["document"] for c in data["citations"]}
            retrieval_ok = q["expect_document"] in cited_docs if q["expect_document"] else len(cited_docs) > 0

        citation_ok = None
        if data["citations"]:
            checks = []
            for c in data["citations"]:
                doc_text = doc_texts.get(c["document"], "")
                checks.append(excerpt_appears_in_doc(c["excerpt"], doc_text))
            citation_ok = all(checks) if checks else None
        elif not q["answerable"]:
            citation_ok = True  # correctly returned no citations

        # Answer correctness is only meaningful for questions we expect an answer to, and
        # only when the question actually declares the fact we're looking for.
        answer_ok = None
        if q["answerable"] and q.get("expect_keywords"):
            answer_ok = answer_matches_keywords(data["answer"], q["expect_keywords"])

        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_answerable": q["answerable"],
                "got_answerable": data["answerable"],
                "confidence": data["confidence"],
                "refusal_ok": refusal_ok,
                "retrieval_ok": retrieval_ok,
                "citation_ok": citation_ok,
                "answer_ok": answer_ok,
                "num_citations": len(data["citations"]),
            }
        )

    metrics = print_report(rows)
    return rows, metrics


def _rate(rows, key):
    """Pass rate over the rows where `key` was actually applicable (not None).

    Returns (rate, passed, applicable). A metric with no applicable rows scores 0.0 rather
    than a vacuous 100% -- "we never checked" must not read as "everything passed".
    """
    applicable = [r for r in rows if r.get(key) is not None]
    passed = sum(1 for r in applicable if r[key])
    rate = passed / len(applicable) if applicable else 0.0
    return rate, passed, len(applicable)


def print_report(rows):
    def mark(v):
        if v is None:
            return "  -  "
        return " PASS" if v else " FAIL"

    print(
        f"{'ID':4} {'Answerable?':11} {'Refusal':7} {'Retrieval':9} {'Citation':9} "
        f"{'Answer':7} Question"
    )
    print("-" * 110)
    for r in rows:
        if "error" in r:
            print(
                f"{r['id']:4} {'ERROR':11} {'FAIL':7} {'-':9} {'-':9} {'-':7} "
                f"{r['question'][:50]}  ({r['error']})"
            )
            continue
        exp = "yes" if r["expected_answerable"] else "no"
        got = "yes" if r["got_answerable"] else "no"
        print(
            f"{r['id']:4} {exp+'->'+got:11} {mark(r['refusal_ok']):7} "
            f"{mark(r['retrieval_ok']):9} {mark(r['citation_ok']):9} "
            f"{mark(r['answer_ok']):7} {r['question'][:50]}"
        )

    print("-" * 110)
    n = len(rows)

    # Refusal accuracy is scored over every question -- an errored question is a failure,
    # not an "not applicable", so it uses the full row count rather than _rate().
    refusal_passed = sum(1 for r in rows if r.get("refusal_ok"))
    refusal_acc = refusal_passed / n if n else 0.0

    retrieval_rate, retrieval_passed, retrieval_n = _rate(rows, "retrieval_ok")
    citation_rate, citation_passed, citation_n = _rate(rows, "citation_ok")
    answer_rate, answer_passed, answer_n = _rate(rows, "answer_ok")

    print(f"Refusal accuracy:     {refusal_acc*100:.0f}%  ({refusal_passed}/{n})")
    print(
        f"Retrieval hit rate:   {retrieval_rate*100:.0f}%  "
        f"({retrieval_passed}/{retrieval_n} answerable questions)"
    )
    print(
        f"Citation correctness: {citation_rate*100:.0f}%  "
        f"({citation_passed}/{citation_n} questions with citations)"
    )
    print(
        f"Answer correctness:   {answer_rate*100:.0f}%  "
        f"({answer_passed}/{answer_n} questions with expected keywords)"
    )

    errors = sum(1 for r in rows if "error" in r)
    if errors:
        print(f"\nERRORS: {errors}/{n} questions failed to get a response from the API.")

    return {
        "refusal_accuracy": refusal_acc,
        "retrieval_hit_rate": retrieval_rate,
        "citation_correctness": citation_rate,
        "answer_correctness": answer_rate,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.yaml"))
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.8,
        help="Exit non-zero if any metric falls below this rate (0-1). Use 0 to report only.",
    )
    args = parser.parse_args()

    rows, metrics = run(args.api_url, Path(args.questions))

    # Gate: any API error is an automatic failure, and every quality metric must clear the
    # threshold. This is what makes the script usable as a CI check rather than a report.
    failures = [
        f"{name} {value*100:.0f}% < {args.fail_under*100:.0f}%"
        for name, value in metrics.items()
        if name != "errors" and value < args.fail_under
    ]
    if metrics["errors"]:
        failures.append(f"{metrics['errors']} question(s) errored")

    if failures:
        print(f"\nFAIL: {'; '.join(failures)}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPASS: all metrics >= {args.fail_under*100:.0f}%")
    sys.exit(0)
