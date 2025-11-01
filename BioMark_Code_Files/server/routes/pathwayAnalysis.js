const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const path = require('path');

// Pathway Analysis Endpoint
router.post('/pathway-analysis', async (req, res) => {
  try {
    const { filePath, selectedClasses } = req.body;

    if (!filePath || !selectedClasses || selectedClasses.length !== 2) {
      return res.status(400).json({
        success: false,
        message: 'Invalid input. Please provide a valid file path and two selected classes.',
      });
    }

    const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, '../services/pathway_analysis.py');
    const python = spawn(pythonCommand, [scriptPath, filePath, ...selectedClasses]);

    let output = '';
    let errorOutput = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });
    

    python.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output.trim());
          console.log('Pathway analysis result:', result);
          if (result.success) {
            return res.status(200).json(result);
          } else {
            return res.status(500).json({
              success: false,
              message: 'Pathway analysis failed.',
              error: result.error,
            });
          }
        } catch (err) {
            console.log("Output parsing error:", output);
            return res.status(500).json({
            success: false,
            message: 'Failed to parse pathway analysis result.',
            error: err.toString(),
          });
        }
      } else {
        return res.status(500).json({
          success: false,
          message: 'Python script failed.',
          error: errorOutput,
        });
      }
    });
  } catch (error) {
    console.error('Error in pathway analysis:', error);
    return res.status(500).json({
      success: false,
      message: 'An error occurred during pathway analysis.',
    });
  }
});

module.exports = router;