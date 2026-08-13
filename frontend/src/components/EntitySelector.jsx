// Entity choice materially changes the answer: the group and subsidiary policies carry
// different figures on the same topics. Leaving it on "infer from question" is the
// honest default — the backend then reads the entity from the question text and falls
// back to group scope — but selecting explicitly is more reliable than phrasing.

export const ENTITIES = [
  { value: "", label: "Infer from question" },
  { value: "Northwind Financial Services Pvt. Ltd.", label: "Northwind Financial Services (group)" },
  { value: "Northwind Capital Markets Ltd", label: "Northwind Capital Markets" },
  { value: "Northwind Payments (Singapore) Pte Ltd", label: "Northwind Payments (Singapore)" },
];

export default function EntitySelector({ entity, onChange, disabled }) {
  return (
    <label className="entity-selector">
      <span className="entity-label">Entity</span>
      <select value={entity} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {ENTITIES.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
