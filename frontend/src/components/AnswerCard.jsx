import CitationList from "./CitationList";
import ConfidenceIndicator from "./ConfidenceIndicator";
import RefusalMessage from "./RefusalMessage";

export default function AnswerCard({ result }) {
  if (!result) return null;

  return (
    <div className="result-box">
      <div className="result-header">
        <ConfidenceIndicator confidence={result.confidence} />
        {!result.grounded && <RefusalMessage refusalReason={result.refusal_reason} />}
      </div>

      <p className="answer-text">{result.answer}</p>

      <CitationList citations={result.citations} />
    </div>
  );
}
