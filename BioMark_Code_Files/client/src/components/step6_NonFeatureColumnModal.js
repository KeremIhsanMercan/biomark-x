import React, { useState, useEffect, useMemo } from 'react';
import '../css/step6_modal.css'; // General CSS for modal

const NonFeatureColumnModal = ({
  isOpen,
  onClose,
  initialColumns, // First 10 columns
  allColumns, // All columns (after fetch)
  selectedColumns, // Selected columns from App.js
  onSelectionChange, // Function to update App.js when selection changes
  onFetchAllColumns, // Function to fetch all columns
  isLoadingAllColumns, // Is fetching all columns?
  error, // Possible errors
}) => {
  // State for search input
  const [searchTerm, setSearchTerm] = useState('');
  // State to track if all columns have been fetched
  const [allColumnsFetched, setAllColumnsFetched] = useState(false);
  // Local state for selected columns
  const [localSelectedColumns, setLocalSelectedColumns] = useState(selectedColumns);

  // Update local state when selectedColumns prop changes
  useEffect(() => {
    setLocalSelectedColumns(selectedColumns);
  }, [selectedColumns]);

  // Update fetch state when allColumns prop changes or modal closes
  useEffect(() => {
    if (allColumns && allColumns.length > 0 && !isLoadingAllColumns) {
      setAllColumnsFetched(true);
    }
    // Reset fetch state and search when modal closes
    if (!isOpen) {
        setAllColumnsFetched(false);
        setSearchTerm('');
    }
  }, [allColumns, isLoadingAllColumns, isOpen]);

  // Handle search input change
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  // Handle checkbox change for column selection
  const handleCheckboxChange = (event) => {
    const columnName = event.target.value;
    const isChecked = event.target.checked;
    let updatedSelection;

    if (isChecked) {
      updatedSelection = [...localSelectedColumns, columnName];
    } else {
      updatedSelection = localSelectedColumns.filter((col) => col !== columnName);
    }
    setLocalSelectedColumns(updatedSelection); // Instantly update local state
    onSelectionChange(updatedSelection); // Notify App.js
  };

  // Handle fetch all columns button click
  const handleFetchAllClick = () => {
    onFetchAllColumns(); // Call fetch function from App.js
  };

  // Determine which columns to display (first 10 or all, filtered by search)
  const displayedColumns = useMemo(() => {
    const columnsToDisplay = allColumnsFetched ? allColumns : initialColumns;
    if (!searchTerm) {
      return columnsToDisplay;
    }
    return columnsToDisplay.filter((col) =>
      col.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [initialColumns, allColumns, allColumnsFetched, searchTerm]);

  // If modal is not open, render nothing
  if (!isOpen) {
    return null;
  }

  return (
    // Modal overlay background
    <div className="modal-overlay">
      {/* Modal content container */}
      <div className="modal-content non-feature-modal">
        {/* Close button */}
        <button className="close-modal-button" onClick={onClose}>×</button>
        <h2>Select Non-Feature Columns</h2>

        {/* Show error message if exists */}
        {error && <div className="error-message">{error}</div>}

        {/* Search bar or fetch all columns button */}
        {!allColumnsFetched && !isLoadingAllColumns && (
          <button
            onClick={handleFetchAllClick}
            className="button fetch-all-columns-button"
          >
            Search for another column
          </button>
        )}

        {/* Loading spinner and message while fetching columns */}
        {isLoadingAllColumns && (
          <div className="loading-message">
             <div className="spinner"></div>
             Columns are fetching...
          </div>
        )}

        {/* Show search input only when all columns are available */}
        {allColumnsFetched && (
          <input
            type="text"
            placeholder="Search columns..."
            value={searchTerm}
            onChange={handleSearchChange}
            className="search-input"
          />
        )}

        {/* Column list with checkboxes */}
        <div className="column-list">
          {displayedColumns.length > 0 ? (
            displayedColumns.map((col, index) => (
              // Each column item with a checkbox
              <div key={index} className="column-item">
                <label>
                  <input
                    type="checkbox"
                    value={col}
                    checked={localSelectedColumns.includes(col)}
                    onChange={handleCheckboxChange}
                  />
                  {col}
                </label>
              </div>
            ))
          ) : (
             // Message if no columns found
             !isLoadingAllColumns && <p>No columns found{searchTerm ? ' matching your search' : ''}.</p>
          )}
        </div>

        {/* Done button to close modal */}
        <div className="modal-actions">
           <button className="button button-primary" onClick={onClose}>
             Done
           </button>
        </div>

      </div>
    </div>
  );
};

export default NonFeatureColumnModal; 