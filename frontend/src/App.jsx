import { useCallback, useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CONFIDENCE_COLORS = {
  high: "#1a7f37",
  medium: "#9a6700",
  low: "#cf222e",
};

function ConfidenceBadge({ confidence }) {
  const color = CONFIDENCE_COLORS[confidence] || "#57606a";
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 10px",
        borderRadius: "999px",
        fontSize: "0.8rem",
        fontWeight: 600,
        color: "white",
        backgroundColor: color,
        textTransform: "uppercase",
      }}
    >
      {confidence} confidence
    </span>
  );
}

function formatTimestamp(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // History is best-effort context, not the main deliverable — if the endpoint is
  // unreachable we leave the list empty rather than surfacing an error that would
  // distract from the answer the user actually asked for.
  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/history?limit=10`);
      if (!res.ok) return;
      setHistory(await res.json());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      loadHistory();
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // Past answers are already stored in full, so replaying one is a local state
  // change — no need to spend another LLM call re-answering the same question.
  function showPastAnswer(item) {
    setQuestion(item.question);
    setError(null);
    setResult({
      answer: item.answer,
      citations: item.citations || [],
      confidence: item.confidence,
      answerable: item.answerable,
    });
  }

  return (
    <div className="page">
      <header>
        <h1>Compliance Copilot</h1>
        <p className="subtitle">
          Ask a question about the sample policy documents. Answers are grounded in the
          documents only — if it's not in there, the tool will say so.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How long do we keep financial records?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <div className="result-box">
          <div className="result-header">
            <ConfidenceBadge confidence={result.confidence} />
            {!result.answerable && <span className="unanswerable-tag">Not answerable from docs</span>}
          </div>

          <p className="answer-text">{result.answer}</p>

          {result.citations.length > 0 && (
            <div className="citations">
              <h3>Sources</h3>
              <ul>
                {result.citations.map((c, i) => (
                  <li key={i} className="citation-item">
                    <div className="citation-source">
                      <strong>{c.document}</strong> — {c.section}
                    </div>
                    <blockquote>{c.excerpt}</blockquote>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="history">
          <h3>Recent questions</h3>
          <ul>
            {history.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className="history-item"
                  onClick={() => showPastAnswer(item)}
                >
                  <span className="history-question">{item.question}</span>
                  <span className="history-meta">
                    <span className={`history-dot history-dot-${item.confidence}`} />
                    {item.answerable ? item.confidence : "not in docs"}
                    {" · "}
                    {formatTimestamp(item.created_at)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
