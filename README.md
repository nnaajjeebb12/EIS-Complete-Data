# Battery Classification System using EIS Data

## Overview

This project implements a deep learning-based classification system for batteries using Electrochemical Impedance Spectroscopy (EIS) data. The system can classify batteries into three categories:

- New
- Used
- Degraded

## Features

- Advanced feature engineering for EIS data processing
- Neural network-based classification model
- Comprehensive battery analysis tools
- Visualization capabilities for EIS data (Nyquist and Bode plots)
- Batch prediction support
- Model persistence and reusability

## Technical Stack

- Python 3.x
- Key Libraries:
  - TensorFlow/Keras (Deep Learning)
  - Pandas (Data Processing)
  - NumPy (Numerical Operations)
  - Scikit-learn (Machine Learning Tools)
  - Matplotlib (Visualization)

## Project Structure

```
├── Training_Data/
│   └── DATA_MODEL.xlsx    # Training dataset
├── Evaluation_Data/
│   ├── new/              # New battery test data
│   ├── used/             # Used battery test data
│   ├── degraded/         # Degraded battery test data
│   └── unknown1/         # Unknown battery test data
├── battery_classifier.keras  # Trained model
├── model_selector.pkl       # Feature selector
├── scaler_battery.pkl       # Data scaler
└── clean_eis_model.ipynb    # Main notebook
```

## Features Engineering

The system implements extensive feature engineering, including:

1. Basic Features:

   - Frequency
   - Real Impedance
   - Imaginary Impedance

2. Enhanced Features:

   - Impedance Magnitude
   - Phase Angle
   - Admittance (Real and Imaginary)
   - Log Transformations
   - Normalized Frequency

3. Statistical Features:

   - Mean and Standard Deviation of Real/Imaginary components
   - Max/Min values
   - Frequency-based slopes
   - Low/High frequency characteristics
   - Band ratios

## Model Architecture

The neural network model consists of:

- Input layer matching feature dimensions
- Dense layers with ReLU activation
- Batch normalization layers
- Dropout layers (0.2) for regularization
- L2 regularization
- Softmax output layer for 3-class classification

## Usage

1. Training New Model:

   ```python
   # Load and run the notebook
   jupyter notebook clean_eis_model.ipynb
   ```

2. Analyzing a Battery:

   ```python
   # Use the predict_from_saved function
   results = predict_from_saved('path/to/battery_file.xlsx')
   ```

3. Batch Processing:

   ```python
   # Run the main function for batch analysis
   main()
   ```

## Output and Visualization

The system generates:

- Nyquist plots
- Bode plots (magnitude and phase)
- Classification results with confidence scores
- Feature importance analysis
- Comparative analysis across battery categories

## Files Generated

- `battery_comparison.png`: Comparison plots for different battery categories
- `example_new_battery.png`: Example analysis for new battery
- `example_used_battery.png`: Example analysis for used battery
- `example_degraded_battery.png`: Example analysis for degraded battery

## Model Performance

The system provides:

- Individual prediction confidence scores
- Batch prediction statistics
- Confusion matrix and classification reports
- Feature importance rankings

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Key Functions

1. `predict_from_saved(file_path)`: Predicts battery state from Excel file
2. `display_battery_analysis(file_path)`: Comprehensive analysis with visualizations
3. `run_batch_predictions()`: Batch processing of evaluation data
4. `display_bode_vectors(data)`: Displays EIS vectors
5. `create_comparison_plots(results)`: Creates comparative visualizations

## Notes

- The system expects Excel files containing EIS data with specific frequency, real, and imaginary components
- Standard data format: 75 measurements per battery
- Model is optimized for three-class classification
- Includes robust error handling and data validation

## Future Improvements

- Extended support for different data formats
- Additional battery health indicators
- Real-time monitoring capabilities
- Enhanced visualization options
- Integration with hardware systems
