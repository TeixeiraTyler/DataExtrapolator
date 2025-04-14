import React, { useState } from 'react';
import './app.css';

function App() {
    const [file, setFile] = useState(null);
    const [dependentVariable, setDependentVariable] = useState('');
    const [numFeatures, setNumFeatures] = useState('');
    const [status, setStatus] = useState({ type: '', message: '' });
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile && selectedFile.type === 'text/csv') {
            setFile(selectedFile);
            setStatus({ type: '', message: '' });
        } else {
            setStatus({ type: 'error', message: 'Please select a valid CSV file' });
            setFile(null);
        }
    };

    const handleDependentVariableChange = (event) => {
        setDependentVariable(event.target.value);
        setStatus({ type: '', message: '' });
    };

    const handleNumFeaturesChange = (event) => {
        setNumFeatures(event.target.value);
        setStatus({ type: '', message: '' });
    };

    const handleUpload = async () => {
        if (!file) {
            setStatus({ type: 'error', message: 'Please select a file first' });
            return;
        }

        if (!dependentVariable.trim()) {
            setStatus({ type: 'error', message: 'Please enter a dependent variable name' });
            return;
        }

        if (!numFeatures.trim()) {
            setStatus({ type: 'error', message: 'Please enter a number of features' });
            return;
        }

        setIsLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('dependent_variable', dependentVariable);
        formData.append('num_features', numFeatures);
        
        try {
            const response = await fetch('http://localhost:8000/api/process-csv', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `Server responded with status: ${response.status}`);
            }

            setResults(data);
            setStatus({ type: 'success', message: 'File processed successfully!' });
            setFile(null);
            setDependentVariable('');
            setNumFeatures(10);
        } catch (error) {
            setStatus({ type: 'error', message: 'Error processing file: ' + error.message });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="container">
            <h1 className="title">Data Extrapolator Tool</h1>
            
            <div className="form-container">
                <div className="form-group">
                    <label className="label">
                        Upload CSV File 
                    </label>
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="file-input"
                    />
                    {file && (
                        <p className="selected-file">
                            Selected file: {file.name}
                        </p>
                    )}
                </div>

                <div className="form-group">
                    <label className="label">
                        Number of Features to Use 
                    </label>
                    <select
                        value={numFeatures}
                        onChange={handleNumFeaturesChange}
                        className="text-input"
                    >
                        <option value="">Select number of features</option>
                        {[...Array(20)].map((_, i) => (
                            <option key={i + 1} value={i + 1}>
                                {i + 1}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="form-group">
                    <label className="label">
                        Dependent Variable Name 
                    </label>
                    <input
                        type="text"
                        value={dependentVariable}
                        onChange={handleDependentVariableChange}
                        placeholder="Enter dependent variable name"
                        className="text-input"
                    />
                </div>

                <button
                    onClick={handleUpload}
                    disabled={isLoading}
                    className="submit-button"
                >
                    {isLoading ? 'Processing...' : 'Upload and Process'}
                </button>

                {status.message && (
                    <div className={`status-message ${status.type}`}>
                        {status.message}
                    </div>
                )}
            </div>

            <div className="additional-components">
                {results && (
                    <div className="results-container">
                        <h2>Analysis Results</h2>
                        
                        <div className="metrics-section">
                            <h3>Model Performance</h3>
                            <div className="metrics-grid">
                                {Object.entries(results.model_scores).map(([model, scores]) => (
                                    <div key={model} className="metric-card">
                                        <h4>{model}</h4>
                                        <p>R² Score: {scores.average_r2.toFixed(3)} ± {scores.std_r2.toFixed(3)}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="metrics-section">
                            <h3>Final Model Metrics</h3>
                            <div className="metrics-grid">
                                {Object.entries(results.final_metrics).map(([metric, value]) => (
                                    <div key={metric} className="metric-card">
                                        <h4>{metric.toUpperCase()}</h4>
                                        <p>{value.toFixed(3)}</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="features-container">
                            <div className="features-section">
                                <h3>Selected Features</h3>
                                <ul className="features-list">
                                    {results.selected_features.map((feature, index) => (
                                        <li key={index}>{feature}</li>
                                    ))}
                                </ul>
                            </div>

                            <div className="features-dropped-section">
                                <h3>Dropped Features</h3>
                                <ul className="features-list">
                                    {results.dropped_columns.map((feature, index) => (
                                        <li key={index}>{feature}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>

                        <div className="plot-section">
                            <h3></h3>
                            <img 
                                src={`data:image/png;base64,${results.correlation_plot}`}
                                alt="Feature Correlations"
                                className="correlation-plot"
                            />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;