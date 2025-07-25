import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import './css/responsive.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Create the root element for the React application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

reportWebVitals();
