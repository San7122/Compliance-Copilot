export default function QuestionInput({ question, onChange, onSubmit, loading }) {
  return (
    <form
      className="query-form"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <input
        type="text"
        value={question}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. How quickly must a data breach be reported?"
        disabled={loading}
      />
      <button type="submit" disabled={loading || !question.trim()}>
        {loading ? "Thinking..." : "Ask"}
      </button>
    </form>
  );
}
