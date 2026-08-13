// Citations are rendered exactly as the backend supplies them. Every field here comes
// from the stored chunk record, so the clause and page shown are the ones a reader can
// actually turn to in the source PDF.

function reference(citation) {
  const parts = [];
  if (citation.document_id) parts.push(citation.document_id);
  if (citation.clause) parts.push(`clause ${citation.clause}`);
  if (citation.page) parts.push(`p. ${citation.page}`);
  return parts.join(" · ");
}

export default function CitationList({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="citations">
      <h3>Sources</h3>
      <ul>
        {citations.map((c) => (
          <li key={c.chunk_id + c.excerpt} className="citation-item">
            <div className="citation-source">
              <strong>{c.document}</strong>
              {reference(c) && <span className="citation-ref"> — {reference(c)}</span>}
            </div>
            {/* The entity matters: subsidiary policies carry different figures from
                the group ones, so an unlabelled citation could be read as group-wide. */}
            {c.entity && <div className="citation-entity">{c.entity}</div>}
            <blockquote>{c.excerpt}</blockquote>
          </li>
        ))}
      </ul>
    </div>
  );
}
