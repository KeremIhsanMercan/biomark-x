const express = require('express');
const bcrypt = require('bcryptjs');
const db = require('../db/database');
const { verifyToken } = require('../middleware/auth');
const fs = require('fs');
const path = require('path');

const router = express.Router();

// Get user statistics
router.get('/stats', verifyToken, (req, res) => {
  try {
    const userId = req.userId;
    
    // Get total uploads
    const uploadsCount = db.prepare('SELECT COUNT(*) as count FROM uploads WHERE user_id = ?').get(userId);
    
    // Get total analyses (both single-upload and merged-file analyses)
    const analysesQuery = db.prepare(`
      SELECT COUNT(*) as count 
      FROM analyses a
      WHERE a.user_id = ?
    `).get(userId);
    
    // Get account creation date
    const account = db.prepare('SELECT created_at FROM accounts WHERE id = ?').get(userId);
    
    return res.json({
      success: true,
      stats: {
        totalUploads: uploadsCount.count,
        totalAnalyses: analysesQuery.count,
        accountCreated: account.created_at
      }
    });
  } catch (err) {
    console.error('Error fetching stats:', err);
    return res.status(500).json({ success: false, message: 'Failed to fetch statistics' });
  }
});

// Get user's analyses
router.get('/analyses', verifyToken, (req, res) => {
  try {
    const userId = req.userId;
    
    const analyses = db.prepare(`
      SELECT 
        a.id,
        a.upload_id,
        a.merged_file_id,
        a.result_path,
        a.status,
        a.created_at,
        a.analysis_metadata,
        u.original_name as filename
      FROM analyses a
      LEFT JOIN uploads u ON a.upload_id = u.id
      WHERE a.user_id = ?
      ORDER BY a.created_at DESC
    `).all(userId);
    
    // Enhance analyses with merged file information and parse metadata
    const enhancedAnalyses = analyses.map(analysis => {
      // Parse analysis metadata
      if (analysis.analysis_metadata) {
        try {
          analysis.metadata = JSON.parse(analysis.analysis_metadata);
        } catch (err) {
          console.error('Error parsing analysis metadata:', err);
          analysis.metadata = null;
        }
      }
      delete analysis.analysis_metadata; // Remove raw JSON string
      
      if (analysis.merged_file_id) {
        // Read metadata file to get source filenames
        const metadataPath = path.join(__dirname, '..', 'results', 'merged_files', `merged_${analysis.merged_file_id}_metadata.json`);
        try {
          if (fs.existsSync(metadataPath)) {
            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            const sourceFiles = Object.keys(metadata.input_files || {});
            analysis.filename = `Merged Files (${sourceFiles.join(', ')})`;
            analysis.isMerged = true;
            analysis.sourceFiles = sourceFiles;
          } else {
            analysis.filename = `Merged file (${analysis.merged_file_id})`;
            analysis.isMerged = true;
          }
        } catch (err) {
          console.error('Error reading merged file metadata:', err);
          analysis.filename = `Merged file (${analysis.merged_file_id})`;
          analysis.isMerged = true;
        }
      } else {
        analysis.isMerged = false;
      }
      return analysis;
    });
    
    return res.json({
      success: true,
      analyses: enhancedAnalyses
    });
  } catch (err) {
    console.error('Error fetching analyses:', err);
    return res.status(500).json({ success: false, message: 'Failed to fetch analyses' });
  }
});

// Get a single analysis by ID
router.get('/analyses/:id', verifyToken, (req, res) => {
  try {
    const userId = req.userId;
    const analysisId = req.params.id;
    
    const analysis = db.prepare(`
      SELECT 
        a.id,
        a.upload_id,
        a.merged_file_id,
        a.result_path,
        a.status,
        a.created_at,
        a.analysis_metadata,
        u.original_name as filename
      FROM analyses a
      LEFT JOIN uploads u ON a.upload_id = u.id
      WHERE a.id = ? AND a.user_id = ?
    `).get(analysisId, userId);
    
    if (!analysis) {
      return res.status(404).json({ success: false, message: 'Analysis not found' });
    }
    
    // Parse analysis metadata
    if (analysis.analysis_metadata) {
      try {
        analysis.metadata = JSON.parse(analysis.analysis_metadata);
      } catch (err) {
        console.error('Error parsing analysis metadata:', err);
        analysis.metadata = null;
      }
    }
    delete analysis.analysis_metadata; // Remove raw JSON string
    
    // Enhance with merged file information
    if (analysis.merged_file_id) {
      const metadataPath = path.join(__dirname, '..', 'results', 'merged_files', `merged_${analysis.merged_file_id}_metadata.json`);
      try {
        if (fs.existsSync(metadataPath)) {
          const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
          const sourceFiles = Object.keys(metadata.input_files || {});
          analysis.filename = `Merged Files (${sourceFiles.join(', ')})`;
          analysis.isMerged = true;
          analysis.sourceFiles = sourceFiles;
        } else {
          analysis.filename = `Merged file (${analysis.merged_file_id})`;
          analysis.isMerged = true;
        }
      } catch (err) {
        console.error('Error reading merged file metadata:', err);
        analysis.filename = `Merged file (${analysis.merged_file_id})`;
        analysis.isMerged = true;
      }
    } else {
      analysis.isMerged = false;
    }
    
    return res.json({
      success: true,
      analysis
    });
  } catch (err) {
    console.error('Error fetching analysis:', err);
    return res.status(500).json({ success: false, message: 'Failed to fetch analysis' });
  }
});

module.exports = router;
