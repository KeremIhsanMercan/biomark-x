// API utility functions for HTTP requests
import axios from 'axios';

// Base URL for API requests, uses environment variable or defaults to localhost
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5003';

// Axios instance for making API calls
export const api = axios.create({
  baseURL: API_BASE
});

// Helper function to build full API endpoint URLs
export const buildUrl = (path) => `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;