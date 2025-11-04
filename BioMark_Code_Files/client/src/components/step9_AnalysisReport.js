import React, { useState, useEffect, useMemo } from 'react';
import jsPDF from 'jspdf';
import '../css/step9-generateAnalysisReport.css';
import { buildUrl } from '../api';

const MAX_PATHWAY_ROWS = 10;

const extractField = (row, candidates) => {
  if (!row || typeof row !== 'object') {
    return undefined;
  }

  for (const key of candidates) {
    if (row[key] !== undefined && row[key] !== null && String(row[key]).length > 0) {
      return row[key];
    }
  }

  return undefined;
};

const formatPValue = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return value !== undefined && value !== null && String(value).length > 0 ? String(value) : 'N/A';
  }
  if (numeric === 0) {
    return '< 1e-4';
  }
  if (numeric >= 0.01) {
    return numeric.toFixed(3);
  }
  return numeric.toExponential(2);
};

const formatNumericValue = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return value !== undefined && value !== null && String(value).length > 0 ? String(value) : 'N/A';
  }
  if (numeric >= 100 || numeric <= 0.01) {
    return numeric.toExponential(2);
  }
  return numeric.toFixed(2);
};

const formatGeneList = (value) => {
  if (!value) {
    return 'N/A';
  }

  if (Array.isArray(value)) {
    const joined = value
      .map((gene) => String(gene).trim())
      .filter(Boolean)
      .join(', ');
    return joined.length > 0 ? joined : 'N/A';
  }

  const normalized = String(value)
    .split(/[;|,]/)
    .map((gene) => gene.trim())
    .filter((gene) => gene.length > 0)
    .join(', ');

  return normalized.length > 0 ? normalized : 'N/A';
};

const preparePathwayRows = (terms = [], limit = MAX_PATHWAY_ROWS) => {
  if (!Array.isArray(terms)) {
    return [];
  }

  return terms
    .filter((row) => row && typeof row === 'object')
    .slice(0, limit)
    .map((row, idx) => {
      const termName =
        extractField(row, ['Term', 'term', 'Pathway', 'pathway', 'Name', 'name']) || `Pathway ${idx + 1}`;
      const overlap = extractField(row, ['Overlap', 'overlap']) || 'N/A';
      const adjustedValue = extractField(row, ['Adjusted P-value', 'Adjusted P-Value', 'AdjustedPValue', 'Adjusted_p_value']);
      const rawValue = extractField(row, ['P-value', 'P-Value', 'pValue', 'PValue']);
      const oddsRatioValue = extractField(row, ['Odds Ratio', 'Odds ratio', 'OddsRatio']);
      const genes = extractField(row, ['Genes', 'genes', 'Gene Names', 'GeneNames', 'Gene names']);

      return {
        index: idx,
        termName,
        overlap,
        adjustedValue,
        rawValue,
        oddsRatioValue,
        genes,
      };
    });
};

const countPathwayRows = (terms = []) => {
  if (!Array.isArray(terms)) {
    return 0;
  }

  return terms.filter((row) => row && typeof row === 'object').length;
};

// Component for generating biomarker analysis report
const AnalysisReport = ({ 
  analysisResults, // This prop should have the enriched structure described above
  analysisDate, 
  executionTime, 
  selectedClasses, // Global - last selected or general context
  selectedIllnessColumn, // Global
  selectedAnalyzes, // Global
  featureCount, // Global
  // selectedClassPair, // already comes from summarizeAnalyses
  summaryImagePath, // This prop is related to summarizeAnalyses and its structure is preserved
  summarizeAnalyses, // This prop's structure is good and preserved
  datasetFileName // Name of the file used in the analysis
}) => {
  // State for loading overlay
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logoDataUrl, setLogoDataUrl] = useState(null);

  // Group analyses by class pairs
  const groupedAnalyses = useMemo(() => {
    if (!analysisResults || !Array.isArray(analysisResults)) return {};
    return analysisResults.reduce((acc, analysis) => {
      // Assume each analysis object has a 'classPair' field.
      const classPairKey = analysis.classPair || 'Unknown Class Pair';
      if (!acc[classPairKey]) {
        acc[classPairKey] = [];
      }
      acc[classPairKey].push(analysis);
      return acc;
    }, {});
  }, [analysisResults]);

  const hasStatisticalSection = Array.isArray(summarizeAnalyses) && summarizeAnalyses.length > 0;

  const hasPathwaySection = useMemo(() => {
    return Object.values(groupedAnalyses).some((analyses) =>
      analyses.some((analysis) => {
        const pathway = analysis.pathway;
        if (!pathway) {
          return false;
        }

        const hasTerms = Array.isArray(pathway.terms) && pathway.terms.length > 0;
        return Boolean(pathway.summary || pathway.csvPath || pathway.error || hasTerms);
      })
    );
  }, [groupedAnalyses]);

  const sectionNumbers = useMemo(() => {
    let number = 1;
    const summary = number;
    number += 1;
    const stat = hasStatisticalSection ? number++ : null;
    const pathway = hasPathwaySection ? number++ : null;
    const analysisCharts = number;

    return { summary, stat, pathway, analysisCharts };
  }, [hasStatisticalSection, hasPathwaySection]);
  
  // Load logo as DataURL for PDF
  useEffect(() => {
    const loadLogo = async () => {
      try {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.src = '/logo192.png';
        
        img.onload = () => {
          // Draw logo to canvas and get DataURL
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, img.width, img.height);
          
          setLogoDataUrl(canvas.toDataURL('image/png'));
        };
        
        img.onerror = () => {
          console.error("Logo could not be loaded");
        };
      } catch (error) {
        console.error("Logo loading error:", error);
      }
    };
    
    loadLogo();
  }, []);
  
  // PDF generation function
  const generatePDF = async () => {
    const reportElement = document.getElementById('analysis-report');
    
    if (!reportElement) {
      console.error('Report element not found');
      return;
    }
    
    // Show loading overlay
    setLoading(true);
    setProgress(5);
    
    try {
      // Calculate content height      
      // Set PDF page size based on content
      const pageWidth = 210; // A4 width (mm)
      const pageHeight = 297; // Standard A4 height (mm). Additional pages will be added automatically.
      // contentHeight * 0.3528
      // Create PDF
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: [pageWidth, pageHeight]
      });
      
      setProgress(10);
      
      // Margin values
      const marginLeft = 20;
      const marginRight = 20;
      
      // Content width
      const contentWidth = pageWidth - marginLeft - marginRight;
      
      // 30mm space for logo and title
      const topMargin = 40;
      
    let yPosition = topMargin;
    let currentSectionNumber = 1;
      
      // ----- COVER TITLE -----
      
      // Add logo
      if (logoDataUrl) {
        try {
          const logoWidth = 50;
          const logoHeight = 50;
          const logoX = (pageWidth - logoWidth) / 2;
          const logoY = yPosition;
          
          pdf.addImage(logoDataUrl, 'PNG', logoX, logoY, logoWidth, logoHeight);
          yPosition += logoHeight + 20;
        } catch (error) {
          console.error("Error adding logo to PDF:", error);
        }
      }
      
      // Report title
      pdf.setFontSize(28);
      pdf.setTextColor(40, 40, 40);
      pdf.setFont('helvetica', 'bold');
      pdf.text('BIOMARKER', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;
      pdf.text('ANALYSIS REPORT', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;
      
      // Decorative line
      pdf.setDrawColor(74, 109, 167);
      pdf.setLineWidth(1);
      pdf.line(marginLeft + 30, yPosition, pageWidth - marginRight - 30, yPosition);
      yPosition += 20;
      
      // Subtitle
      pdf.setFontSize(16);
      pdf.setTextColor(80, 80, 80);
      pdf.setFont('helvetica', 'italic');
      pdf.text('Comprehensive Analysis Results', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;
      
      // Class info - List all analyzed pairs
      pdf.setFontSize(12); // Adjusted font size
      pdf.setTextColor(90, 90, 90);
      pdf.setFont('helvetica', 'normal');
      
      if (Object.keys(groupedAnalyses).length > 0) {
        Object.keys(groupedAnalyses).forEach(classPair => {
          if (yPosition > pageHeight - 50) { // New page if near end
            pdf.addPage();
            yPosition = topMargin - 20;
          }
          pdf.text(`Comparing: ${classPair}`, pageWidth / 2, yPosition, { align: 'center' });
          yPosition += 8;
        });
      } else if (selectedClasses && selectedClasses.length >= 2) {
        // Fallback to global selectedClasses if no groupedAnalyses
        pdf.text(`Comparing: ${selectedClasses.join(' vs ')}`, pageWidth / 2, yPosition, { align: 'center' });
        yPosition += 8;
      }
      yPosition += 12;
      
      // Decorative bottom line
      pdf.setDrawColor(220, 220, 220);
      pdf.setLineWidth(0.5);
      pdf.line(marginLeft + 40, yPosition, pageWidth - marginRight - 40, yPosition);
      yPosition += 20;
      
      // Corporate info
      pdf.setFontSize(10);
      pdf.setTextColor(150, 150, 150);
      pdf.text('Biomarker Analysis Tool © ' + new Date().getFullYear(), pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 10;
      pdf.text('All Rights Reserved', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 30;
      
      // ----- ANALYSIS SUMMARY -----
      
      // Section title
      pdf.setFontSize(16);
      pdf.setTextColor(60, 60, 60);
      pdf.setFont('helvetica', 'bold');
  pdf.text(`${currentSectionNumber}. Analysis Summary`, marginLeft, yPosition);
      yPosition += 10;
      
      // Bottom line
      pdf.setDrawColor(74, 109, 167);
      pdf.setLineWidth(0.5);
      pdf.line(marginLeft, yPosition, marginLeft + 50, yPosition);
      yPosition += 15;
      
      // Summary info - now by grouped analyses
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      pdf.setTextColor(80, 80, 80);
      
      const leftColumnX = marginLeft;
      const lineHeight = 6;

      // Dataset filename info
      if (datasetFileName) {
        pdf.setFont('helvetica', 'bold');
        pdf.text('Dataset Filename:', leftColumnX, yPosition);
        pdf.setFont('helvetica', 'normal');
        pdf.text(datasetFileName, leftColumnX + 40, yPosition);
        yPosition += lineHeight + 5;
      }

      if (Object.keys(groupedAnalyses).length > 0) {
        let groupIndex = 0;
        for (const [classPair, analysesInGroup] of Object.entries(groupedAnalyses)) {
          if (yPosition > pageHeight - 70) { pdf.addPage(); yPosition = topMargin - 20; }
          
          pdf.setFontSize(12);
          pdf.setFont('helvetica', 'bold');
          pdf.setTextColor(65, 65, 65);
          pdf.text(classPair, leftColumnX, yPosition);
          yPosition += lineHeight + 2;
          pdf.setDrawColor(150,150,150);
          pdf.setLineWidth(0.2);
          pdf.line(leftColumnX, yPosition, pageWidth - marginRight, yPosition);
          yPosition += lineHeight + 2;

          let analysisIndexInGroup = 0;
          for (const analysis of analysesInGroup) {
            if (yPosition > pageHeight - 60) { pdf.addPage(); yPosition = topMargin - 20; }
            
            pdf.setFontSize(11);
            pdf.setFont('helvetica', 'bold');
            pdf.setTextColor(70, 70, 70);
            // analysis.title (e.g., "Analysis 1") should already include this.
            // If analysis.title is missing: `Analysis ${analysisIndexInGroup + 1}`
            pdf.text(analysis.title ? `${analysis.title.replace(/Analysis \d+/, `Analysis ${analysisIndexInGroup + 1}`)}` : `Analysis ${analysisIndexInGroup + 1}`, leftColumnX + 5, yPosition);
            yPosition += lineHeight;

            pdf.setFontSize(10);
            pdf.setFont('helvetica', 'normal');
            pdf.setTextColor(80, 80, 80);

            // Analysis date
            pdf.setFont('helvetica', 'bold');
            pdf.text('Analysis Date:', leftColumnX + 10, yPosition);
            pdf.setFont('helvetica', 'normal');
            pdf.text(analysis.date || 'N/A', leftColumnX + 40, yPosition);
            yPosition += lineHeight;

            // Analysis types
            pdf.setFont('helvetica', 'bold');
            pdf.text('Analysis Types:', leftColumnX + 10, yPosition);
            pdf.setFont('helvetica', 'normal');
            let analysisTypesText = 'N/A';
            if (analysis.types && typeof analysis.types === 'object') {
              const types = [];
              if (analysis.types.differential && analysis.types.differential.length) 
                types.push('Differential: ' + analysis.types.differential.join(', '));
              if (analysis.types.clustering && analysis.types.clustering.length) 
                types.push('Clustering: ' + analysis.types.clustering.join(', '));
              if (analysis.types.classification && analysis.types.classification.length) 
                types.push('Classification: ' + analysis.types.classification.join(', '));
              analysisTypesText = types.length ? types.join('; ') : 'N/A';
            }
            const splitTypes = pdf.splitTextToSize(analysisTypesText, contentWidth - 30);
            pdf.text(splitTypes, leftColumnX + 40, yPosition);
            yPosition += lineHeight * splitTypes.length;
            
            // Execution time
            pdf.setFont('helvetica', 'bold');
            pdf.text('Execution Time:', leftColumnX + 10, yPosition);
            pdf.setFont('helvetica', 'normal');
            pdf.text(analysis.time || 'N/A', leftColumnX + 40, yPosition);
            yPosition += lineHeight + 5;
            analysisIndexInGroup++;
          }
          
          if (groupIndex < Object.keys(groupedAnalyses).length - 1) {
             yPosition += 5;
          }
          groupIndex++;
        }
      } else {
        // Fallback if no groupedAnalyses (old global info)
        pdf.setFont('helvetica', 'bold');
        pdf.text('Analysis Date:', leftColumnX, yPosition);
        pdf.setFont('helvetica', 'normal');
        pdf.text(analysisDate || 'N/A', leftColumnX + 30, yPosition);
        yPosition += lineHeight;
      }
      yPosition += 10;

      currentSectionNumber += 1;

      // ----- STATISTICAL ANALYSIS RESULTS -----
      if (hasStatisticalSection) {
        // Section title
        pdf.setFontSize(16);
        pdf.setTextColor(60, 60, 60);
        pdf.setFont('helvetica', 'bold');
        pdf.text(`${currentSectionNumber}. Statistical Method Results`, marginLeft, yPosition);
        yPosition += 10;
        
        // Bottom line
        pdf.setDrawColor(74, 109, 167);
        pdf.setLineWidth(0.5);
        pdf.line(marginLeft, yPosition, marginLeft + 70, yPosition);
        yPosition += 15;
        
        // Add summary image - summarizeAnalyses already comes by classPair
        for (let k = 0; k < summarizeAnalyses.length; k++) {
          const summaryAnalysis = summarizeAnalyses[k];
          if (yPosition > pageHeight - 80) { pdf.addPage(); yPosition = topMargin - 20; }

          pdf.setFontSize(12);
          pdf.setFont('helvetica', 'bold');
          pdf.setTextColor(70, 70, 70);
          pdf.text(`Summary for: ${summaryAnalysis.classPair || 'All Classes'}`, marginLeft, yPosition);
          yPosition += 8;

          try {
            // Load the summary image directly (avoid html2canvas and DOM dependency)
            const img = new Image();
            img.crossOrigin = 'Anonymous';
            // Use buildUrl to construct proper URL with base URL
            img.src = summaryAnalysis.imagePath.startsWith('http') 
              ? summaryAnalysis.imagePath 
              : buildUrl(`/${summaryAnalysis.imagePath}`);

            await new Promise((resolve, reject) => {
              img.onload = () => resolve();
              img.onerror = () => reject(new Error(`Failed to load image: ${summaryAnalysis.imagePath.split('/').pop()}`));
              setTimeout(() => reject(new Error('Image loading timeout')), 15000);
            });

            const canvas = document.createElement('canvas');
            const scaleFactor = 2;
            canvas.width = img.width * scaleFactor;
            canvas.height = img.height * scaleFactor;
            const ctx = canvas.getContext('2d');
            ctx.scale(scaleFactor, scaleFactor);
            ctx.drawImage(img, 0, 0, img.width, img.height);

            const imgData = canvas.toDataURL('image/jpeg', 0.85);
            const aspectRatio = img.width / img.height;
            let imgWidth = contentWidth;
            let imgHeight = imgWidth / aspectRatio;

            const maxImgHeight = pageHeight * 0.6;
            if (imgHeight > maxImgHeight) {
              imgHeight = maxImgHeight;
              imgWidth = imgHeight * aspectRatio;
            }

            if (yPosition + imgHeight > pageHeight - 30) {
              pdf.addPage();
              yPosition = topMargin - 20;
              pdf.setFontSize(12);
              pdf.setFont('helvetica', 'bold');
              pdf.setTextColor(70, 70, 70);
              pdf.text(`Summary for: ${summaryAnalysis.classPair || 'All Classes'} (Continued)`, marginLeft, yPosition);
              yPosition += 8;
            }

            pdf.addImage(imgData, 'JPEG', marginLeft + (contentWidth - imgWidth) / 2, yPosition, imgWidth, imgHeight);
            yPosition += imgHeight + 15;
          } catch (error) {
            console.error('Error adding image:', error);
            if (yPosition > pageHeight - 30) { pdf.addPage(); yPosition = topMargin - 20; }
            pdf.setFontSize(10);
            pdf.setTextColor(255, 0, 0);
            pdf.text(`*Summary image for ${summaryAnalysis.classPair} failed: ${error.message}`, marginLeft, yPosition);
            yPosition += 10;
          }
          yPosition += 10;
        }
        currentSectionNumber += 1;
      }
      
      if (hasPathwaySection) {
        if (yPosition > pageHeight - 40) { pdf.addPage(); yPosition = topMargin - 20; }

        pdf.setFontSize(16);
        pdf.setTextColor(60, 60, 60);
        pdf.setFont('helvetica', 'bold');
        pdf.text(`${currentSectionNumber}. Pathway Analysis`, marginLeft, yPosition);
        yPosition += 10;

        pdf.setDrawColor(74, 109, 167);
        pdf.setLineWidth(0.5);
        pdf.line(marginLeft, yPosition, marginLeft + 65, yPosition);
        yPosition += 15;

        for (const [classPair, analysesInGroup] of Object.entries(groupedAnalyses)) {
          const analysesWithPathway = analysesInGroup.filter((analysis) => analysis.pathway);
          if (!analysesWithPathway.length) {
            continue;
          }

          if (yPosition > pageHeight - 60) { pdf.addPage(); yPosition = topMargin - 20; }

          pdf.setFontSize(13);
          pdf.setFont('helvetica', 'bold');
          pdf.setTextColor(70, 70, 70);
          pdf.text(classPair, marginLeft, yPosition);
          yPosition += lineHeight;

          pdf.setDrawColor(190, 190, 190);
          pdf.setLineWidth(0.3);
          pdf.line(marginLeft, yPosition, pageWidth - marginRight, yPosition);
          yPosition += lineHeight;

          analysesWithPathway.forEach((analysis, idxWithinPathway) => {
            if (yPosition > pageHeight - 60) { pdf.addPage(); yPosition = topMargin - 20; }

            pdf.setFontSize(11);
            pdf.setFont('helvetica', 'bold');
            pdf.setTextColor(75, 75, 75);
            const analysisTitle = analysis.title
              ? analysis.title.replace(/Analysis \d+/, `Analysis ${idxWithinPathway + 1}`)
              : `Analysis ${idxWithinPathway + 1}`;
            pdf.text(`${analysisTitle} Pathway Summary`, marginLeft + 5, yPosition);
            yPosition += lineHeight;

            pdf.setFontSize(10);
            pdf.setFont('helvetica', 'normal');
            pdf.setTextColor(80, 80, 80);

            if (analysis.pathway?.summary) {
              const summaryLines = pdf.splitTextToSize(analysis.pathway.summary, contentWidth - 15);
              pdf.text(summaryLines, marginLeft + 10, yPosition);
              yPosition += lineHeight * summaryLines.length;
            }

            if (analysis.pathway?.error) {
              pdf.setFont('helvetica', 'italic');
              pdf.setTextColor(180, 40, 40);
              const errorLines = pdf.splitTextToSize(`Note: ${analysis.pathway.error}`, contentWidth - 15);
              pdf.text(errorLines, marginLeft + 10, yPosition);
              yPosition += lineHeight * errorLines.length;
              pdf.setFont('helvetica', 'normal');
              pdf.setTextColor(80, 80, 80);
            }

            const metrics = [];
            if (analysis.pathway?.inputGeneCount !== null && analysis.pathway?.inputGeneCount !== undefined) {
              metrics.push(`Input genes: ${analysis.pathway.inputGeneCount}`);
            }
            if (analysis.pathway?.totalPathways !== null && analysis.pathway?.totalPathways !== undefined) {
              metrics.push(`Total pathways: ${analysis.pathway.totalPathways}`);
            }
            if (analysis.pathway?.significantPathwayCount !== null && analysis.pathway?.significantPathwayCount !== undefined) {
              metrics.push(`Significant (<= 0.05): ${analysis.pathway.significantPathwayCount}`);
            }

            if (metrics.length > 0) {
              const metricsLines = pdf.splitTextToSize(metrics.join(' \u2022 '), contentWidth - 15);
              pdf.text(metricsLines, marginLeft + 10, yPosition);
              yPosition += lineHeight * metricsLines.length;
            }

            const topRows = preparePathwayRows(analysis.pathway?.terms, MAX_PATHWAY_ROWS);

            if (topRows.length > 0) {
              const columns = [
                { key: 'index', header: '#', width: 8, align: 'center' },
                { key: 'termName', header: 'Pathway', width: 50, align: 'left' },
                { key: 'overlap', header: 'Overlap', width: 18, align: 'center' },
                { key: 'adjusted', header: 'Adjusted p-value', width: 24, align: 'center' },
                { key: 'raw', header: 'Raw p-value', width: 22, align: 'center' },
                { key: 'odds', header: 'Odds ratio', width: 18, align: 'center' },
                { key: 'genes', header: 'Genes', width: 30, align: 'left' }
              ];
              const baseLineHeight = 4.2;
              const cellPadding = 1.5;
              const headerFontSize = 10;
              const bodyFontSize = 9;

              const formattedRows = topRows.map((row) => {
                const genesFormatted = formatGeneList(row.genes);
                const geneParts = genesFormatted === 'N/A' ? [] : genesFormatted.split(', ');
                const truncatedGenes = genesFormatted === 'N/A'
                  ? 'N/A'
                  : geneParts.slice(0, 10).join(', ') + (geneParts.length > 10 ? ', ...' : '');

                return {
                  index: String(row.index + 1),
                  termName: row.termName,
                  overlap: row.overlap || 'N/A',
                  adjusted: formatPValue(row.adjustedValue),
                  raw: formatPValue(row.rawValue),
                  odds: formatNumericValue(row.oddsRatioValue),
                  genes: truncatedGenes
                };
              });

              pdf.setFont('helvetica', 'bold');
              pdf.setFontSize(headerFontSize);
              const headerLines = columns.map((column) => {
                const lines = pdf.splitTextToSize(column.header, column.width - cellPadding * 2);
                return lines.length ? lines : [column.header];
              });
              const headerHeight = Math.max(
                ...headerLines.map((lines) => Math.max(lines.length, 1))
              ) * baseLineHeight + cellPadding * 2;
              const tableWidth = columns.reduce((sum, col) => sum + col.width, 0);

              const drawHeader = () => {
                if (yPosition + headerHeight > pageHeight - 30) {
                  pdf.addPage();
                  yPosition = topMargin - 20;
                }
                const headerTop = yPosition;
                const tableLeft = marginLeft + 5;
                let columnX = tableLeft;
                pdf.setFillColor(230, 235, 246);
                pdf.setDrawColor(190, 190, 190);
                pdf.setLineWidth(0.2);
                pdf.rect(tableLeft, headerTop, tableWidth, headerHeight, 'F');
                pdf.setFont('helvetica', 'bold');
                pdf.setFontSize(headerFontSize);
                columns.forEach((column, idx) => {
                  const lines = headerLines[idx];
                  pdf.rect(columnX, headerTop, column.width, headerHeight);
                  const textStartY = headerTop + cellPadding + baseLineHeight;
                  if (column.align === 'center') {
                    const centerX = columnX + column.width / 2;
                    lines.forEach((line, lineIdx) => {
                      pdf.text(line, centerX, textStartY + lineIdx * baseLineHeight, { align: 'center' });
                    });
                  } else {
                    lines.forEach((line, lineIdx) => {
                      pdf.text(line, columnX + cellPadding, textStartY + lineIdx * baseLineHeight, { align: 'left' });
                    });
                  }
                  columnX += column.width;
                });
                yPosition += headerHeight;
                pdf.setFont('helvetica', 'normal');
                pdf.setFontSize(bodyFontSize);
              };

              drawHeader();

              formattedRows.forEach((row) => {
                const splitTexts = columns.map((column) => {
                  const cellValue = row[column.key];
                  return pdf.splitTextToSize(cellValue, column.width - cellPadding * 2);
                });

                const rowHeight = Math.max(
                  ...splitTexts.map((lines) => Math.max(lines.length, 1))
                ) * baseLineHeight + cellPadding * 2;

                if (yPosition + rowHeight > pageHeight - 30) {
                  pdf.addPage();
                  yPosition = topMargin - 20;
                  drawHeader();
                }

                let cellX = marginLeft + 5;
                columns.forEach((column, idx) => {
                  const cellHeight = rowHeight;
                  pdf.rect(cellX, yPosition, column.width, cellHeight);
                  const lines = splitTexts[idx];
                  const textStartY = yPosition + cellPadding + 3;
                  if (column.align === 'center') {
                    const centerX = cellX + column.width / 2;
                    lines.forEach((line, lineIdx) => {
                      pdf.text(line, centerX, textStartY + lineIdx * baseLineHeight, {
                        align: 'center',
                      });
                    });
                  } else {
                    lines.forEach((line, lineIdx) => {
                      pdf.text(line, cellX + cellPadding, textStartY + lineIdx * baseLineHeight, {
                        align: 'left',
                      });
                    });
                  }
                  cellX += column.width;
                });

                yPosition += rowHeight;
              });

              yPosition += 4;
            }

            const totalRows = countPathwayRows(analysis.pathway?.terms);
            if (analysis.pathway?.terms && totalRows > topRows.length) {
              pdf.setFont('helvetica', 'italic');
              pdf.setTextColor(110, 110, 110);
              pdf.text(
                `Additional ${totalRows - topRows.length} pathways are available in the CSV file.`,
                marginLeft + 10,
                yPosition
              );
              yPosition += lineHeight;
              pdf.setFont('helvetica', 'normal');
              pdf.setTextColor(80, 80, 80);
            }

            if (analysis.pathway?.csvPath) {
              const csvLink = analysis.pathway.csvPath.startsWith('http')
                ? analysis.pathway.csvPath
                : buildUrl(`/${analysis.pathway.csvPath}`);
              pdf.setFont('helvetica', 'italic');
              pdf.setTextColor(60, 60, 140);
              if (typeof pdf.textWithLink === 'function') {
                pdf.textWithLink('Download', marginLeft + 10, yPosition, { url: csvLink });
              } else {
                pdf.text('Download', marginLeft + 10, yPosition);
              }
              yPosition += lineHeight;
              pdf.setFont('helvetica', 'normal');
              pdf.setTextColor(80, 80, 80);
            }

            yPosition += 4;
          });

          yPosition += 6;
        }

        currentSectionNumber += 1;
      }

      // ----- DETAILED ANALYSIS RESULTS (Charts) -----
      if (Object.keys(groupedAnalyses).length > 0) {
        if (yPosition > pageHeight - 40) { pdf.addPage(); yPosition = topMargin - 20; }
        // Section title
        pdf.setFontSize(16);
        pdf.setTextColor(60, 60, 60);
        pdf.setFont('helvetica', 'bold');
        pdf.text(`${currentSectionNumber}. Analysis Results`, marginLeft, yPosition);
        yPosition += 10;
        
        // Bottom line
        pdf.setDrawColor(74, 109, 167);
        pdf.setLineWidth(0.5);
        pdf.line(marginLeft, yPosition, marginLeft + 85, yPosition);
        yPosition += 15;
        
        let groupIdxForResults = 0;
        for (const [classPair, analysesInGroup] of Object.entries(groupedAnalyses)) {
          if (yPosition > pageHeight - 60) { pdf.addPage(); yPosition = topMargin - 20; }

          pdf.setFontSize(14);
          pdf.setFont('helvetica', 'bold');
          pdf.setTextColor(65, 65, 65);
          pdf.text(classPair, marginLeft, yPosition);
          yPosition += 8;
           pdf.setDrawColor(180,180,180);
           pdf.setLineWidth(0.2);
           pdf.line(marginLeft, yPosition, pageWidth - marginRight, yPosition);
           yPosition += 10;

          let analysisIdxInResults = 0;
          for (const analysis of analysesInGroup) {
            if (yPosition > pageHeight - 50) { pdf.addPage(); yPosition = topMargin - 20; }
            
            pdf.setFontSize(12);
            pdf.setTextColor(70, 70, 70);
            pdf.setFont('helvetica', 'bold');
            // analysis.title (e.g., "Analysis 1") should already include this.
            pdf.text(analysis.title ? `${analysis.title.replace(/Analysis \d+/, `Analysis ${analysisIdxInResults + 1}`)} for ${classPair}` : `Analysis ${analysisIdxInResults + 1} for ${classPair}`, marginLeft + 5, yPosition);
            yPosition += 8;
            
            if (analysis.images && analysis.images.length > 0) {
              for (let j = 0; j < analysis.images.length; j++) {
                try {
                  if (analysis.images[j].path) {
                    if (yPosition > pageHeight - 80 && !(j === 0 && analysisIdxInResults === 0 && groupIdxForResults === 0)) {
                       pdf.addPage(); 
                       yPosition = topMargin - 20; 
                    }

                    // Image caption
                    if (analysis.images[j].caption) {
                      pdf.setFontSize(10);
                      pdf.setTextColor(100, 100, 100);
                      pdf.setFont('helvetica', 'italic');
                      const splitCaption = pdf.splitTextToSize(analysis.images[j].caption, contentWidth);
                      pdf.text(splitCaption, marginLeft + 5, yPosition);
                      yPosition += 5 * splitCaption.length;
                    }
                    
                    const img = new Image();
                    img.crossOrigin = "Anonymous";
                    // Use buildUrl to construct proper URL with base URL
                    img.src = analysis.images[j].path.startsWith('http') 
                      ? analysis.images[j].path 
                      : buildUrl(`/${analysis.images[j].path}`);
                    
                    await new Promise((resolve, reject) => {
                      img.onload = () => {
                        resolve();
                      };
                      img.onerror = (err) => {
                        reject(new Error(`Failed to load image: ${analysis.images[j].path.split('/').pop()}`));
                      };
                      setTimeout(() => {
                        reject(new Error('Image loading timeout'));
                      }, 15000);
                    });
                    
                    const canvas = document.createElement('canvas');
                    const scaleFactor = 2;
                    canvas.width = img.width * scaleFactor;
                    canvas.height = img.height * scaleFactor;
                    const ctx = canvas.getContext('2d');
                    ctx.scale(scaleFactor, scaleFactor);
                    ctx.drawImage(img, 0, 0, img.width, img.height);
                    
                    const imgData = canvas.toDataURL('image/jpeg', 0.85);
                    const aspectRatio = img.width / img.height;
                    let imgPdfWidth = contentWidth;
                    let imgPdfHeight = imgPdfWidth / aspectRatio;

                    // Adjust image size to prevent page overflow
                    const maxImgHeight = pageHeight * 0.7;
                    if (imgPdfHeight > maxImgHeight) {
                        imgPdfHeight = maxImgHeight;
                        imgPdfWidth = imgPdfHeight * aspectRatio;
                    }
                    if (imgPdfWidth > contentWidth) {
                        imgPdfWidth = contentWidth;
                        imgPdfHeight = imgPdfWidth / aspectRatio;
                    }

                    if (yPosition + imgPdfHeight > pageHeight - 25) {
                      pdf.addPage();
                      yPosition = topMargin - 20;
                       pdf.setFontSize(10);
                       pdf.setTextColor(100,100,100);
                       pdf.setFont('helvetica', 'italic');
                       pdf.text(analysis.images[j].caption + " (Continued)", marginLeft+5, yPosition);
                       yPosition +=5;
                    }
                    
                    pdf.addImage(imgData, 'JPEG', marginLeft + (contentWidth - imgPdfWidth) / 2, yPosition, imgPdfWidth, imgPdfHeight);
                    yPosition += imgPdfHeight + 10;
                  }
                } catch (error) {
                  if (yPosition > pageHeight - 30) { pdf.addPage(); yPosition = topMargin - 20; }
                  pdf.setFontSize(9);
                  pdf.setTextColor(255, 0, 0);
                  pdf.text(`*Image '${analysis.images[j].caption}' could not be loaded: ${error.message}`, marginLeft + 5, yPosition);
                  yPosition += 5;
                }
              }
            }
            yPosition += 5;
            analysisIdxInResults++;
          }
          if (groupIdxForResults < Object.keys(groupedAnalyses).length - 1) {
            yPosition += 10;
            if (yPosition > pageHeight - 30) { pdf.addPage(); yPosition = topMargin - 20; }
            pdf.setDrawColor(200,200,200);
            pdf.setLineWidth(0.3);
            pdf.line(marginLeft, yPosition, pageWidth-marginRight, yPosition);
            yPosition += 10;
          }
          groupIdxForResults++;
        }
      }
      
      // Footer
      pdf.setFontSize(8);
      pdf.setTextColor(150, 150, 150);
      pdf.setFont('helvetica', 'italic');
      const currentDate = new Date().toLocaleString();
      const version = "1.0.0";
      
      // Leave enough space for footer
      yPosition += 5;
      
      // Footer line
      pdf.setDrawColor(200, 200, 200);
      pdf.setLineWidth(0.5);
      pdf.line(marginLeft, yPosition, pageWidth - marginRight, yPosition);
      yPosition += 15;
      
      // Footer text
      pdf.text(`This report was automatically generated by Biomarker Analysis Tool v${version} on ${currentDate}`, pageWidth / 2, yPosition, { align: 'center' });
      
      // Save PDF
      pdf.save(`Biomarker_Analysis_Report_${new Date().toISOString().split('T')[0]}_${datasetFileName}.pdf`);
      
      setProgress(100);
      
      // Hide loading overlay after completion
      setTimeout(() => {
        setLoading(false);
        setProgress(0);
      }, 500);
    } catch (error) {
      setLoading(false);
      setProgress(0);
      alert('An error occurred while generating the report. Please try again.');
    }
  };

  // Version info
  const version = "1.0.0";

  return (
    <div>
      <button 
        className="generate-report-button" 
        onClick={generatePDF}
        title="Generate a professional PDF report of your analysis results"
        disabled={loading}
      >
        <i className="report-icon">{loading ? '➳' : '📊'}</i>
        {loading ? 'Generating Report...' : 'Generate Analysis Report'}
      </button>
      
      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            Generating your professional report... ({progress}%)
          </div>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}
      
      {/* Hidden report template - html2canvas will be used for PDF generation */}
      <div id="analysis-report" className="hidden-report-template">
        <div className="report-content">
          {/* Cover Title */}
          <div className="report-header">
            <h1>BIOMARKER ANALYSIS REPORT</h1>
            <h2>Comprehensive Analysis Results</h2>
            {Object.keys(groupedAnalyses).length > 0 ? (
              Object.keys(groupedAnalyses).map(classPair => (
                <p key={classPair}>Comparing: {classPair}</p>
              ))
            ) : (
              selectedClasses && selectedClasses.length >= 2 && (
                <p>Comparing: {selectedClasses.join(' vs ')}</p>
              )
            )}
          </div>

          {/* Analysis Summary */}
          <div className="report-section">
            <h3>{sectionNumbers.summary}. Analysis Summary</h3>
            {datasetFileName && (
              <div className="info-row">
                <span className="label">Dataset Filename:</span>
                <span className="value">{datasetFileName}</span>
              </div>
            )}
            {Object.keys(groupedAnalyses).length > 0 ? (
              Object.entries(groupedAnalyses).map(([classPair, analysesInGroup]) => (
                <div key={classPair} className="class-pair-summary-group">
                  <h4>{classPair}</h4>
                  {analysesInGroup.map((analysis, index) => (
                    <div key={analysis.title || index} className="analysis-summary-item">
                      <h5>{analysis.title ? analysis.title.replace(/Analysis \d+/, `Analysis ${index + 1}`) : `Analysis ${index + 1}`}</h5>
                      <div className="info-row">
                        <span className="label">Analysis Date:</span>
                        <span className="value">{analysis.date || 'N/A'}</span>
                      </div>
                      <div className="info-row">
                        <span className="label">Analysis Types:</span>
                        <span className="value">
                          {(() => {
                            if (analysis.types && typeof analysis.types === 'object') {
                              const types = [];
                              if (analysis.types.differential?.length) 
                                types.push('Differential: ' + analysis.types.differential.join(', '));
                              if (analysis.types.clustering?.length) 
                                types.push('Clustering: ' + analysis.types.clustering.join(', '));
                              if (analysis.types.classification?.length) 
                                types.push('Classification: ' + analysis.types.classification.join(', '));
                              return types.length ? types.join('; ') : 'N/A';
                            }
                            return 'N/A';
                          })()}
                        </span>
                      </div>
                      <div className="info-row">
                        <span className="label">Execution Time:</span>
                        <span className="value">{analysis.time || 'N/A'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ))
            ) : (
              <div className="summary-info"> {/* Fallback to old global summary if no grouped data */}
                 <div className="info-row">
                    <span className="label">Analysis Date:</span>
                    <span className="value">{analysisDate || 'N/A'}</span>
                  </div>
                  {/* ... other global fields ... */}
              </div>
            )}
          </div>

          {/* Statistical Analysis Results */}
          {hasStatisticalSection && (
            <div className="report-section">
              <h3>{sectionNumbers.stat}. Statistical Method Results</h3>
              {summarizeAnalyses.map((analysis, index) => (
                // Add data-classpair to help PDF image selector
                <div key={index} className="summary-section" data-classpair={analysis.classPair}>
                  <h4>Analysis for {analysis.classPair}</h4>
                  <div className="summary-image">
                    <img 
                        src={analysis.imagePath.startsWith('http') ? analysis.imagePath : buildUrl(`/${analysis.imagePath}`)} 
                        alt={`Statistical Analysis for ${analysis.classPair}`} 
                        crossOrigin="anonymous"
                    />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Pathway Analysis Results */}
          {hasPathwaySection && (
            <div className="report-section">
              <h3>{sectionNumbers.pathway}. Pathway Analysis</h3>
              {Object.entries(groupedAnalyses).map(([classPair, analysesInGroup]) => {
                const analysesWithPathway = analysesInGroup.filter((analysis) => analysis.pathway);
                if (!analysesWithPathway.length) {
                  return null;
                }

                return (
                  <div key={classPair} className="pathway-report-group">
                    <h4>{classPair}</h4>
                    {analysesWithPathway.map((analysis, index) => {
                      const topRows = preparePathwayRows(analysis.pathway?.terms, MAX_PATHWAY_ROWS);
                      const totalRows = countPathwayRows(analysis.pathway?.terms);
                      const csvHref = analysis.pathway?.csvPath
                        ? (analysis.pathway.csvPath.startsWith('http')
                            ? analysis.pathway.csvPath
                            : buildUrl(`/${analysis.pathway.csvPath}`))
                        : null;

                      return (
                        <div key={analysis.title || index} className="pathway-report-item pathway-results-card">
                          <h5>{analysis.title ? analysis.title.replace(/Analysis \d+/, `Analysis ${index + 1}`) : `Analysis ${index + 1}`}</h5>
                          {analysis.pathway?.summary && (
                            <p className="pathway-summary-text">{analysis.pathway.summary}</p>
                          )}
                          {analysis.pathway?.error && (
                            <p className="pathway-error-state pathway-inline-error">{analysis.pathway.error}</p>
                          )}
                          <div className="pathway-results-metrics">
                            {analysis.pathway?.inputGeneCount !== null && analysis.pathway?.inputGeneCount !== undefined && (
                              <span className="pathway-metric-pill">Input genes: {analysis.pathway.inputGeneCount}</span>
                            )}
                            {analysis.pathway?.totalPathways !== null && analysis.pathway?.totalPathways !== undefined && (
                              <span className="pathway-metric-pill">Total pathways: {analysis.pathway.totalPathways}</span>
                            )}
                            {analysis.pathway?.significantPathwayCount !== null && analysis.pathway?.significantPathwayCount !== undefined && (
                              <span className="pathway-metric-pill">Significant (&le; 0.05): {analysis.pathway.significantPathwayCount}</span>
                            )}
                          </div>
                          {csvHref && (
                            <p className="pathway-download">
                              <a
                                className="pathway-download-link"
                                href={csvHref}
                                target="_blank"
                                rel="noopener noreferrer"
                                aria-label="Download full pathway results as CSV"
                              >
                                Download
                              </a>
                            </p>
                          )}
                          {topRows.length > 0 && (
                            <div className="pathway-table-wrapper">
                              <table className="pathway-results-table">
                                <thead>
                                  <tr>
                                    <th>#</th>
                                    <th>Pathway</th>
                                    <th>Overlap</th>
                                    <th>Adjusted p-value</th>
                                    <th>Raw p-value</th>
                                    <th>Odds ratio</th>
                                    <th>Genes</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {topRows.map((row) => {
                                    const genesFormatted = formatGeneList(row.genes);
                                    const geneParts = genesFormatted === 'N/A' ? [] : genesFormatted.split(', ');
                                    const truncatedGenes = genesFormatted === 'N/A'
                                      ? 'N/A'
                                      : geneParts.slice(0, 12).join(', ') + (geneParts.length > 12 ? ', ...' : '');

                                    return (
                                      <tr key={`${row.termName}-${row.index}`}>
                                        <td>{row.index + 1}</td>
                                        <td className="pathway-term">{row.termName}</td>
                                        <td>{row.overlap || 'N/A'}</td>
                                        <td>{formatPValue(row.adjustedValue)}</td>
                                        <td>{formatPValue(row.rawValue)}</td>
                                        <td>{formatNumericValue(row.oddsRatioValue)}</td>
                                        <td className="pathway-genes">{truncatedGenes}</td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          )}
                          {analysis.pathway?.terms && totalRows > topRows.length && (
                            <p className="pathway-note">
                              Additional {totalRows - topRows.length} pathways are available in the downloaded CSV.
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}

          {/* Detailed Analysis Results (Charts) */}
          {Object.keys(groupedAnalyses).length > 0 && (
            <div className="report-section">
              <h3>{sectionNumbers.analysisCharts}. Analysis Results</h3>
              {Object.entries(groupedAnalyses).map(([classPair, analysesInGroup]) => (
                <div key={classPair} className="class-pair-results-group">
                  <h4>{classPair}</h4>
                  {analysesInGroup.map((analysis, index) => (
                    <div key={analysis.title || index} className="analysis-result-item">
                      <h5>{analysis.title ? analysis.title.replace(/Analysis \d+/, `Analysis ${index + 1}`) : `Analysis ${index + 1}`}</h5>
                      {analysis.images?.map((image, imgIndex) => (
                        <div key={image.id || imgIndex} className="result-image">
                          {image.caption && <p className="image-caption">{image.caption}</p>}
                          <img 
                            src={image.path.startsWith('http') ? image.path : buildUrl(`/${image.path}`)} 
                            alt={image.caption || `Image ${imgIndex + 1} for ${analysis.title}`} 
                            crossOrigin="anonymous"
                           />
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}

          {/* Footer */}
          <div className="report-footer">
            <p>This report was automatically generated by Biomarker Analysis Tool v{version} on {new Date().toLocaleString()}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisReport; 