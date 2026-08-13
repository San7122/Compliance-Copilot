// A refusal is a real outcome, not an error state. It is shown as an answer with an
// explanation of *why* the system declined, because "no evidence was retrieved" and
// "the quotes could not be verified" mean quite different things to the reader.

const REASONS = {
  no_relevant_evidence:
    "Nothing in the policy documents was close enough to this question to rely on.",
  model_declined:
    "Related text was found, but it does not actually answer this question.",
  citations_unverified:
    "A draft answer was produced, but its supporting quotes could not be verified against the source documents, so it was withheld.",
};

export default function RefusalMessage({ refusalReason }) {
  return (
    <div className="refusal">
      <span className="unanswerable-tag">Not answerable from the documents</span>
      {REASONS[refusalReason] && <p className="refusal-detail">{REASONS[refusalReason]}</p>}
    </div>
  );
}
