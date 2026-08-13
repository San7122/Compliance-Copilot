// All backend access lives here. Components never call fetch directly, and there is no
// RAG logic on the client -- the frontend's only job is to present what the API returns.

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_URL}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function askQuestion(question, entity) {
  return request("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // `entity` is omitted when blank so the backend infers it from the question.
    body: JSON.stringify(entity ? { question, entity } : { question }),
  });
}

export function fetchHistory(limit = 10) {
  return request(`/api/history?limit=${limit}`);
}
