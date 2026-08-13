// Confidence is a score in [0,1], not a probability. The label is deliberately
// qualitative and the numeric value is shown alongside it rather than instead of it, so
// the UI doesn't imply a precision the score doesn't have.

const BANDS = [
  { min: 0.75, label: "high", color: "#1a7f37" },
  { min: 0.45, label: "medium", color: "#9a6700" },
  { min: 0.0, label: "low", color: "#cf222e" },
];

export function confidenceBand(confidence) {
  const value = typeof confidence === "number" ? confidence : 0;
  return BANDS.find((band) => value >= band.min) ?? BANDS[BANDS.length - 1];
}

export default function ConfidenceIndicator({ confidence }) {
  const band = confidenceBand(confidence);
  const percent = Math.round((confidence ?? 0) * 100);

  return (
    <span className="confidence" title="Application-level score, not a calibrated probability">
      <span
        style={{
          display: "inline-block",
          padding: "2px 10px",
          borderRadius: "999px",
          fontSize: "0.8rem",
          fontWeight: 600,
          color: "white",
          backgroundColor: band.color,
          textTransform: "uppercase",
        }}
      >
        {band.label} confidence
      </span>
      <span className="confidence-value"> {percent}%</span>
    </span>
  );
}
