// API utility functions for HTTP requests
import axios from 'axios';
import { getSessionId, setSessionId } from './utils/session';

// Base URL for API requests, uses environment variable or defaults to localhost
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5003';

// Axios instance for making API calls
export const api = axios.create({
  baseURL: API_BASE
});

// Attach session ID to every request
api.interceptors.request.use((config) => {
  config.headers = config.headers || {};
  config.headers['x-session-id'] = getSessionId();
  return config;
});

// Persist new session IDs issued by the backend
api.interceptors.response.use((response) => {
  const incomingId = response.headers['x-session-id'];
  if (incomingId) {
    setSessionId(incomingId);
  }
  return response;
});

// Helper function to build full API endpoint URLs
export const buildUrl = (path) => `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;

// Thin wrapper around the browser fetch API that automatically
// attaches the session header so non-axios calls stay in the same session.
export async function apiFetch(input, init = {}) {
  const sessionId = getSessionId();
  const headers = { ...(init.headers || {}), 'x-session-id': sessionId };
  const response = await fetch(input, { ...init, headers });
  // Persist potential updated session id from the server
  const newId = response.headers.get('x-session-id');
  if (newId) {
    setSessionId(newId);
  }
  return response;
}