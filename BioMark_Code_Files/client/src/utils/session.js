export function getSessionId() {
  let id = localStorage.getItem('session_id');
  if (!id) {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      id = crypto.randomUUID();
    } else {
      // Fallback (not RFC-4122 compliant but guarantees uniqueness for demo)
      id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    localStorage.setItem('session_id', id);
  }
  return id;
}

export function setSessionId(id) {
  if (id) {
    localStorage.setItem('session_id', id);
  }
} 