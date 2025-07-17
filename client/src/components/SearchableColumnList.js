import React, { useState, useEffect, useMemo } from 'react';
import '../css/SearchableColumnList.css'; // Import style file

function SearchableColumnList({
  initialColumns = [],
  allColumns = [],
  onSelect,
  selectedColumns = [], // For multi-select or to highlight single selection
  placeholder = "Search columns...",
  listHeight = '200px', // Adjustable list height
  className = '', // For adding extra class
  disabled = false, // To disable the list
}) {
  // State for the search input value
  const [searchTerm, setSearchTerm] = useState('');

  // Filter all columns according to the search term
  const filteredColumns = useMemo(() => {
    if (!searchTerm) {
      return initialColumns; // Show initial columns if there is no search
    }
    const lowerCaseSearchTerm = searchTerm.toLowerCase();
    return allColumns.filter(col =>
      col.toLowerCase().includes(lowerCaseSearchTerm)
    );
  }, [searchTerm, initialColumns, allColumns]);

  // Handle click event for a column item
  const handleColumnClick = (column) => {
    if (!disabled && onSelect) {
      onSelect(column);
    }
  };

  // Convert selected columns to a Set for faster lookup
  const selectedSet = useMemo(() => new Set(Array.isArray(selectedColumns) ? selectedColumns : [selectedColumns]), [selectedColumns]);

  return (
    // Main container for the searchable column list
    <div className={`searchable-column-list-container ${className}`}>
      {/* Search input for filtering columns */}
      <input
        type="text"
        placeholder={placeholder}
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="column-search-input"
        disabled={disabled || allColumns.length === 0} // Disable input if no columns or component is disabled
      />
      {/* Info text if there are no columns and not disabled */}
      {allColumns.length === 0 && !disabled && <p className="info-text-small">Loading columns or no columns available.</p>}
      {/* List of columns, filtered by search term */}
      <ul className="column-list" style={{ maxHeight: listHeight }}>
        {filteredColumns.length > 0 ? (
          filteredColumns.map((column, index) => (
            // Each column item in the list
            <li
              key={index}
              onClick={() => handleColumnClick(column)}
              className={`
                column-list-item
                ${selectedSet.has(column) ? 'selected' : ''}
                ${disabled ? 'disabled' : ''}
              `}
              title={column} // Add title for long names
            >
              {column}
            </li>
          ))
        ) : (
          // Message if no columns match the search term
          searchTerm && <li className="column-list-item-no-results">No matching columns found.</li>
        )}
      </ul>
    </div>
  );
}

export default SearchableColumnList;