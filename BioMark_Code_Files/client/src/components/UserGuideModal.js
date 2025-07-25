import React from 'react';
import '../css/userGuideModal.css';

const UserGuideModal = ({ onClose }) => {
  return (
    <div className="user-guide-overlay">
      <div className="user-guide-modal">
        <div className="popup-header">
          <h2>Biomarker Analysis Tool - User Guide</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        <div className="popup-content">
          <p className="guide-description">
            This tool enables researchers to explore expression datasets to discover potential biomarkers. Upload your data, configure the analysis pipeline, and generate comprehensive visual and statistical reports in just a few clicks.
          </p>
          <div className="video-container">
            {/* TODO: Replace the `src` with the actual introduction video URL */}
            <iframe
              src="https://www.youtube.com/@dizi_sahnem/featured"
              title="Biomarker Analysis Tool Introduction"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserGuideModal; 