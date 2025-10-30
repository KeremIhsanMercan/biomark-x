import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './index.css';
import './css/responsive.css';
import App from './App';
import LoginPage from './pages/LoginPage';
import ProfilePage from './pages/ProfilePage';
import AnalysisResultsPage from './pages/AnalysisResultsPage';
import AnalysisDetailPage from './pages/AnalysisDetailPage';
import reportWebVitals from './reportWebVitals';

// Protected Route Component
function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" replace />;
}

// Create the root element for the React application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <Router basename="/biomark">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <App />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/profile" 
          element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/my-analyses" 
          element={
            <ProtectedRoute>
              <AnalysisResultsPage />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/analysis/:analysisId" 
          element={
            <ProtectedRoute>
              <AnalysisDetailPage />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </Router>
  </React.StrictMode>
);

reportWebVitals();
