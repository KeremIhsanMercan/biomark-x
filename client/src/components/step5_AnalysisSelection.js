import React, { useState, useRef, useEffect } from 'react';

function AnalysisSelection({ onAnalysisSelection }) {
  // State for selected analysis method and category
  const [selectedMethod, setSelectedMethod] = useState({
    category: null,
    method: null,
  });
  // State for button and parameter dropdown
  const [buttonPressed, setButtonPressed] = useState(false);
  const [showParamsDropdown, setShowParamsDropdown] = useState(false);
  const [useDefaultParams, setUseDefaultParams] = useState(true);
  const [paramsChanged, setParamsChanged] = useState(false);
  const [confirmSelection, setConfirmSelection] = useState(false);
  
  // Ref for parameter settings section (for scrolling)
  const parameterSettingsRef = useRef(null);
  
  // Parameter states
  // Differential Analysis Parameters
  const [featureType, setFeatureType] = useState("microRNA");
  const [referenceClass, setReferenceClass] = useState("");
  const [limeGlobalExplanationSampleNum, setLimeGlobalExplanationSampleNum] = useState(50);
  const [shapModelFinetune, setShapModelFinetune] = useState(false);
  const [limeModelFinetune, setLimeModelFinetune] = useState(false);
  const [scoring, setScoring] = useState("f1");
  const [featureimportanceFinetune, setFeatureimportanceFinetune] = useState(false);
  const [numTopFeatures, setNumTopFeatures] = useState(20);
  
  // Clustering Analysis Parameters
  const [plotter, setPlotter] = useState("seaborn");
  const [dim, setDim] = useState("3D");
  
  // Classification Analysis Parameters
  const [paramFinetune, setParamFinetune] = useState(false);
  const [finetuneFraction, setFinetuneFraction] = useState(1.0);
  const [saveBestModel, setSaveBestModel] = useState(true);
  const [standardScaling, setStandardScaling] = useState(true);
  const [saveDataTransformer, setSaveDataTransformer] = useState(true);
  const [saveLabelEncoder, setSaveLabelEncoder] = useState(true);
  const [verbose, setVerbose] = useState(true);
  
  // Common parameters
  const [testSize, setTestSize] = useState(0.2);
  const [nFolds, setNFolds] = useState(5);
  
  // Scroll to parameter settings when shown
  useEffect(() => {
    // When parameter settings become visible, scroll to them
    if (showParamsDropdown && confirmSelection && parameterSettingsRef.current) {
      setTimeout(() => {
        // Offset for header/banner height
        const headerHeight = document.querySelector('.app-header')?.offsetHeight || 0;
        const yOffset = -headerHeight - 20;
        const element = parameterSettingsRef.current;
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        window.scrollTo({
          top: elementPosition + yOffset,
          behavior: 'smooth'
        });
      }, 300);
    }
  }, [showParamsDropdown, confirmSelection]);
  
  const differentialAnalysis = ['SHAP', 'LIME', 'Anova', 'T_test', 'Xgb-Feature-Importance', 'Randomforest-Feature-Importance', 'Permutation-Feature-Importance'];
  const clusteringAnalysis = ['PCA', 'tSNE', 'UMAP'];
  const classificationAnalysis = ['Logistic Regression', 'Random Forest', 'XGBClassifier', 'Decision Tree', 'Gradient Boosting', 'CatBoosting Classifier', 'AdaBoost Classifier', 'MLPClassifier', 'SVC'];

  // Handle selection of analysis method
  const handleSelection = (method, category) => {
    if (selectedMethod.category === category && selectedMethod.method === method) {
      // Deselect if the same method is clicked
      setSelectedMethod({ category: null, method: null });
      setShowParamsDropdown(false);
      setConfirmSelection(false);
    } else {
      // Update selection
      setSelectedMethod({ category, method });
      setShowParamsDropdown(false);
      setConfirmSelection(false);
    }
  };
  
  // Handle confirm selection button click
  const handleConfirmSelection = () => {
    setButtonPressed(true);
    setShowParamsDropdown(true);
    setConfirmSelection(true);
    // Scroll to parameter settings after render
    setTimeout(() => {
      if (parameterSettingsRef.current) {
        const headerHeight = document.querySelector('.app-header')?.offsetHeight || 0;
        const yOffset = -headerHeight - 20;
        const element = parameterSettingsRef.current;
        const elementPosition = element.getBoundingClientRect().top + window.pageYOffset;
        window.scrollTo({
          top: elementPosition + yOffset,
          behavior: 'smooth'
        });
      }
    }, 300);
  }
  
  // Track parameter changes
  const handleParamChange = () => {
    setParamsChanged(true);
    setUseDefaultParams(false);
  };

  // Update parameter settings
  const handleUpdateParams = () => {
    setParamsChanged(false);
    completeSelection();
  };

  // Use default parameter settings
  const handleUseDefaultParams = () => {
    setUseDefaultParams(true);
    setParamsChanged(false);
    // Reset all parameters to default values
    setFeatureType("microRNA");
    setReferenceClass("");
    setLimeGlobalExplanationSampleNum(50);
    setShapModelFinetune(false);
    setLimeModelFinetune(false);
    setScoring("f1");
    setFeatureimportanceFinetune(false);
    setNumTopFeatures(20);
    setPlotter("seaborn");
    setDim("3D");
    setParamFinetune(false);
    setFinetuneFraction(1.0);
    setSaveBestModel(true);
    setStandardScaling(true);
    setSaveDataTransformer(true);
    setSaveLabelEncoder(true);
    setVerbose(true);
    setTestSize(0.2);
    setNFolds(5);
    completeSelection();
  };
  
  // Send selection and parameters to parent component
  const completeSelection = () => {
    const result = {
      differential: selectedMethod.category === 'differential' ? [selectedMethod.method] : [],
      clustering: selectedMethod.category === 'clustering' ? [selectedMethod.method] : [],
      classification: selectedMethod.category === 'classification' ? [selectedMethod.method] : [],
      useDefaultParams: useDefaultParams,
      parameters: {
        featureType,
        referenceClass,
        limeGlobalExplanationSampleNum,
        shapModelFinetune,
        limeModelFinetune,
        scoring,
        featureimportanceFinetune,
        numTopFeatures,
        plotter,
        dim,
        paramFinetune,
        finetuneFraction,
        saveBestModel,
        standardScaling,
        saveDataTransformer,
        saveLabelEncoder,
        verbose,
        testSize,
        nFolds
      }
    };
    onAnalysisSelection(result);
  };

  return (
    <div className="analysis-selection">
      <div className='analysis-tables'>
        {/* Differential Factor Analysis table */}
        <div className="analysis-category">
          <h4>Differential Factor Analysis</h4>
          <table>
            <tbody>
              {differentialAnalysis.map((method, index) => (
                <tr
                  key={index}
                  className={selectedMethod.category === 'differential' && selectedMethod.method === method ? 'selected' : ''}
                  onClick={() => {handleSelection(method, 'differential');}}
                >
                  <td>{method}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Clustering Analysis table */}
        <div className="analysis-category">
          <h4>Clustering Analysis</h4>
          <table>
            <tbody>
              {clusteringAnalysis.map((method, index) => (
                <tr
                  key={index}
                  className={selectedMethod.category === 'clustering' && selectedMethod.method === method ? 'selected' : ''}
                  onClick={() => {handleSelection(method, 'clustering');}}
                >
                  <td>{method}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Classification Analysis table */}
        <div className="analysis-category">
          <h4>Classification Analysis</h4>
          <table>
            <tbody>
              {classificationAnalysis.map((method, index) => (
                <tr
                  key={index}
                  className={selectedMethod.category === 'classification' && selectedMethod.method === method ? 'selected' : ''}
                  onClick={() => {handleSelection(method, 'classification');}}
                >
                  <td>{method}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
        
      {/* Confirm analysis selection button */}
      {!confirmSelection && (
      <div className='analysis-button'>
        <button
            onClick={handleConfirmSelection}
            disabled={!selectedMethod.category || !selectedMethod.method} // Disabled until a selection is made
          >
            Confirm Selection
          </button>
      </div>
      )}
      
      {/* Parameter Settings section - shown after Confirm Selection */}
      {confirmSelection && showParamsDropdown && (
        <div className="parameter-settings" ref={parameterSettingsRef}>
          <h4>Parameter Settings</h4>
          <div className="selected-method-info">
            Selected Method: {selectedMethod.method}
          </div>
          
          <div className="param-container">
            {/* Differential Analysis Parameters */}
            {selectedMethod.category === 'differential' && (
              <div className="param-section">
                {/* Each parameter row is explained with a tooltip */}
                <div className="param-row">
                  <div className="param-label">
                    feature_type
                    <span className="param-tooltip">Specifies the type of features used in the analysis (e.g., microRNA, gene, protein, etc.)</span>
                  </div>
                  <div className="param-input">
                    <input 
                      type="text" 
                      value={featureType} 
                      onChange={(e) => {
                        setFeatureType(e.target.value);
                        handleParamChange();
                      }} 
                    />
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    reference_class
                    <span className="param-tooltip">The reference class used for comparison in differential analysis. Leave empty to use default.</span>
                  </div>
                  <div className="param-input">
                    <input 
                      type="text" 
                      value={referenceClass} 
                      onChange={(e) => {
                        setReferenceClass(e.target.value);
                        handleParamChange();
                      }} 
                    />
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    lime_global_explanation_sample_num
                    <span className="param-tooltip">Number of samples to use for LIME global explanation. Higher values provide more stable explanations but take longer to compute.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={limeGlobalExplanationSampleNum}
                      onChange={(e) => {
                        setLimeGlobalExplanationSampleNum(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(num => (
                        <option key={num} value={num}>{num}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    shap_model_finetune
                    <span className="param-tooltip">When enabled, fine-tunes the model used for SHAP explanations. This may improve accuracy but increases computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={shapModelFinetune.toString()}
                      onChange={(e) => {
                        setShapModelFinetune(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    lime_model_finetune
                    <span className="param-tooltip">When enabled, fine-tunes the model used for LIME explanations. This may improve accuracy but increases computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={limeModelFinetune.toString()}
                      onChange={(e) => {
                        setLimeModelFinetune(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    scoring
                    <span className="param-tooltip">Metric used to evaluate model performance. Options include: f1 (balanced), recall (sensitivity), precision (positive predictive value), and accuracy.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={scoring}
                      onChange={(e) => {
                        setScoring(e.target.value);
                        handleParamChange();
                      }}
                    >
                      <option value="f1">f1</option>
                      <option value="recall">recall</option>
                      <option value="precision">precision</option>
                      <option value="accuracy">accuracy</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    featureimportance_finetune
                    <span className="param-tooltip">When enabled, fine-tunes models used for feature importance calculations. Improves accuracy but increases computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={featureimportanceFinetune.toString()}
                      onChange={(e) => {
                        setFeatureimportanceFinetune(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    num_top_features
                    <span className="param-tooltip">Number of top features to select and display in the results. Higher values include more features but may include less important ones.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={numTopFeatures}
                      onChange={(e) => {
                        setNumTopFeatures(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map(num => (
                        <option key={num} value={num}>{num}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    test_size
                    <span className="param-tooltip">Proportion of the dataset to use for testing. Typically 0.2 (20%) is a good balance between training and testing data.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={testSize}
                      onChange={(e) => {
                        setTestSize(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    n_folds
                    <span className="param-tooltip">Number of folds for cross-validation. Higher values provide more robust model evaluation but increase computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={nFolds}
                      onChange={(e) => {
                        setNFolds(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(fold => (
                        <option key={fold} value={fold}>{fold}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            {/* Clustering Analysis Parameters */}
            {selectedMethod.category === 'clustering' && (
              <div className="param-section">
                <div className="param-row">
                  <div className="param-label">
                    plotter
                    <span className="param-tooltip">Visualization library to use for clustering plots. Seaborn offers more aesthetically pleasing plots, while matplotlib provides more customization options.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={plotter}
                      onChange={(e) => {
                        setPlotter(e.target.value);
                        handleParamChange();
                      }}
                    >
                      <option value="seaborn">seaborn</option>
                      <option value="matplotlib">matplotlib</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    dim
                    <span className="param-tooltip">Dimension of the visualization: 2D (flat) or 3D (spatial). 3D may reveal more complex relationships but can be harder to interpret.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={dim}
                      onChange={(e) => {
                        setDim(e.target.value);
                        handleParamChange();
                      }}
                    >
                      <option value="2D">2D</option>
                      <option value="3D">3D</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    test_size
                    <span className="param-tooltip">Proportion of the dataset to use for testing. Typically 0.2 (20%) is a good balance between training and testing data.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={testSize}
                      onChange={(e) => {
                        setTestSize(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    n_folds
                    <span className="param-tooltip">Number of folds for cross-validation. Higher values provide more robust model evaluation but increase computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={nFolds}
                      onChange={(e) => {
                        setNFolds(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(fold => (
                        <option key={fold} value={fold}>{fold}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            {/* Classification Analysis Parameters */}
            {selectedMethod.category === 'classification' && (
              <div className="param-section">
                {/* Each parameter row is explained with a tooltip */}
                <div className="param-row">
                  <div className="param-label">
                    param_finetune
                    <span className="param-tooltip">When enabled, automatically optimizes model hyperparameters, which can improve performance but significantly increases computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={paramFinetune.toString()}
                      onChange={(e) => {
                        setParamFinetune(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    finetune_fraction
                    <span className="param-tooltip">Fraction of parameter space to explore during hyperparameter optimization. Higher values are more thorough but take longer.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={finetuneFraction}
                      onChange={(e) => {
                        setFinetuneFraction(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map(frac => (
                        <option key={frac} value={frac}>{frac}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    save_best_model
                    <span className="param-tooltip">When enabled, saves the best-performing model to disk for future use or further analysis.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={saveBestModel.toString()}
                      onChange={(e) => {
                        setSaveBestModel(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    standard_scaling
                    <span className="param-tooltip">When enabled, standardizes features by removing the mean and scaling to unit variance. Recommended for most models.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={standardScaling.toString()}
                      onChange={(e) => {
                        setStandardScaling(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    save_data_transformer
                    <span className="param-tooltip">When enabled, saves the data scaling/transformation object for consistent preprocessing in future analyses.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={saveDataTransformer.toString()}
                      onChange={(e) => {
                        setSaveDataTransformer(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    save_label_encoder
                    <span className="param-tooltip">When enabled, saves the label encoding object, which maps class names to numerical values and is needed for consistent predictions.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={saveLabelEncoder.toString()}
                      onChange={(e) => {
                        setSaveLabelEncoder(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    verbose
                    <span className="param-tooltip">When enabled, provides detailed output during model training and evaluation, useful for debugging or understanding the process.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={verbose.toString()}
                      onChange={(e) => {
                        setVerbose(e.target.value === "true");
                        handleParamChange();
                      }}
                    >
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    test_size
                    <span className="param-tooltip">Proportion of the dataset to use for testing. Typically 0.2 (20%) is a good balance between training and testing data.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={testSize}
                      onChange={(e) => {
                        setTestSize(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="param-row">
                  <div className="param-label">
                    n_folds
                    <span className="param-tooltip">Number of folds for cross-validation. Higher values provide more robust model evaluation but increase computation time.</span>
                  </div>
                  <div className="param-input">
                    <select 
                      value={nFolds}
                      onChange={(e) => {
                        setNFolds(Number(e.target.value));
                        handleParamChange();
                      }}
                    >
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(fold => (
                        <option key={fold} value={fold}>{fold}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Parameter buttons */}
          <div className="param-buttons">
            <div className="default-param-option">
              <button onClick={handleUseDefaultParams}>
                Use default parameter settings
              </button>
            </div>
            
            {paramsChanged && (
              <div className="param-update-button">
                <button onClick={handleUpdateParams}>
                  Update Parameter Settings
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Info message for selection */}
      {buttonPressed && !confirmSelection && (
          <div className="info-message">
            <p>Selected Method: {selectedMethod.method}</p>
          </div>
        )}
    </div>
  );
}

export default AnalysisSelection;
