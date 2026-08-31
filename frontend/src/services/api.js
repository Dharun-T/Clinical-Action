const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  } catch {
    throw new Error("FastAPI is not running. Start the backend on http://127.0.0.1:8000, then refresh this page.");
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Backend request failed (${response.status}).`);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  tests: () => request("/tests"),
  patients: () => request("/patients"),
  patient: (id) => request(`/patient/${id}`),
  labs: (id, test) => request(`/labs/${id}?test_name=${encodeURIComponent(test)}`),
  analytics: () => request("/analytics"),
  researchResults: () => request("/research-results"),
  predict: (payload) => request("/predict", { method: "POST", body: JSON.stringify(payload) }),
};