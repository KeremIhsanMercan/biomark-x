import React, { useState, useEffect, useMemo } from 'react';
import '../css/step6_modal.css'; // Reuse existing modal CSS for now, might need adjustments
import '../css/step3_ColumnSearchModal.css'; // Add specific CSS for single selection

const SingleColumnSelectModal = ({
  isOpen,
  onClose,
  title, // Title for the modal
  initialColumns, // Initial columns (e.g., first 10)
  allColumns, // All columns (after fetch)
  selectedColumn, // The currently selected column (single value)
  onSelect, // Function to call when a column is selected
  onFetchAllColumns, // Function to fetch all columns
  isLoadingAllColumns, // Is fetching in progress?
  error, // Potential errors
}) => {
  // State for the search input value
  const [searchTerm, setSearchTerm] = useState('');
  // State to track if all columns have been fetched
  const [allColumnsFetched, setAllColumnsFetched] = useState(false);

  // Reset state when modal opens or closes, or when allColumns change status
  useEffect(() => {
    if (isOpen) {
      // Reset fetch status if allColumns are not available yet when opening
      if (!allColumns || allColumns.length === 0) {
          setAllColumnsFetched(false);
      } else {
          setAllColumnsFetched(true); // Assume fetched if allColumns is provided on open
      }
      setSearchTerm(''); // Clear search on open
    } else {
      // Optionally reset fetch status when closing
      // setAllColumnsFetched(false);
      // setSearchTerm('');
    }
  }, [isOpen, allColumns]);

  // Update fetched status when allColumns prop changes after initial load
   useEffect(() => {
    if (allColumns && allColumns.length > 0 && !isLoadingAllColumns) {
      setAllColumnsFetched(true);
    }
     // Consider if resetting is needed when allColumns becomes empty/null
   }, [allColumns, isLoadingAllColumns]);

  // Handle search input change
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

  // Handle clicking on a column item for selection
  const handleColumnClick = (columnName) => {
    onSelect(columnName); // Pass the selected column name back
    onClose(); // Close modal after selection
  };

  // Handle fetching all columns when button is clicked
  const handleFetchAllClick = () => {
    onFetchAllColumns(); // Call the fetch function provided by App.js
  };

  // Determine which columns to display based on fetch status and search term
  const displayedColumns = useMemo(() => {
    const columnsToDisplay = allColumnsFetched ? allColumns : (initialColumns || []);
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
      <div className="modal-content single-column-modal"> {/* Add specific class */}
        {/* Close button */}
        <button className="close-modal-button" onClick={onClose}>×</button>
        {/* Modal title */}
        <h2>{title || 'Select a Column'}</h2> {/* Use title prop */}

        {/* Error message display */}
        {error && <div className="error-message">{error}</div>}

        {/* Button to fetch all columns or Search Input */}
        {!allColumnsFetched && !isLoadingAllColumns && (initialColumns && initialColumns.length > 0) && ( // Show fetch button only if initial columns are present but not all
          <button
            onClick={handleFetchAllClick}
            className="button fetch-all-columns-button"
            disabled={isLoadingAllColumns} // Disable while loading
          >
            {isLoadingAllColumns ? 'Loading...' : 'Search from All Columns'}
          </button>
        )}

        {/* Loading spinner and message while fetching all columns */}
        {isLoadingAllColumns && (
          <div className="loading-message">
             <div className="spinner"></div>
             Loading all columns...
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
            autoFocus // Auto-focus on search when available
          />
        )}

        {/* Column List for Single Selection */}
        <div className="column-list single-select-list"> {/* Add specific class */}
          {displayedColumns.length > 0 ? (
            displayedColumns.map((col, index) => (
              // Each column item for selection
              <div
                key={index}
                className={`column-item single-select-item ${selectedColumn === col ? 'selected' : ''}`} // Highlight selected
                onClick={() => handleColumnClick(col)} // Handle click for selection
              >
                {col}
              </div>
            ))
          ) : (
             // Message if no columns found
             !isLoadingAllColumns && <p>No columns found{searchTerm ? ' matching your search' : ''}.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default SingleColumnSelectModal; 