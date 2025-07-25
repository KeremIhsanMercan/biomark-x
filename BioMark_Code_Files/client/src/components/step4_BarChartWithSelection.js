import React, { useState } from 'react';

function BarChartWithSelection({ chartUrl, classList, onClassSelection }) {
  // State to keep track of selected classes
  const [selectedClasses, setSelectedClasses] = useState([]);

  // Handle click on a class row to select/deselect
  const handleClassClick = (className) => {
    // Toggle selection
    if (selectedClasses.includes(className)) {
      setSelectedClasses(selectedClasses.filter((cls) => cls !== className));
    } else if (selectedClasses.length < 2) {
      setSelectedClasses([...selectedClasses, className]);
    }
  };

  // Check if a class is selected
  const isSelected = (className) => selectedClasses.includes(className);

  return (
    <div className="bar-chart-with-selection">
      {/* Bar chart image section */}
      <div className="chart-container">
        <img src={chartUrl} alt="Diagnosis Bar Chart" className="chart-image" />
      </div>

      {/* Class selection table section */}
      <div className="class-selection">
        <h3>Classes in Your File</h3>
        <table>
          <tbody>
            {classList.map((className, index) => (
              // Each row represents a class, can be selected
              <tr
                key={index}
                className={isSelected(className) ? 'selected' : ''}
                onClick={() => handleClassClick(className)}
              >
                <td>{className}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {/* Show analyze button only when two classes are selected */}
        {selectedClasses.length === 2 && (
          <button onClick={() => onClassSelection(selectedClasses)}>Analyze</button>
        )}
      </div>
    </div>
  );
}

export default BarChartWithSelection;
