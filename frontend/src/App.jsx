import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:5000/api';

export default function App() {
  // Navigation & UI state
  const [navScrolled, setNavScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  // Detection state
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [dragOver, setDragOver] = useState(false);

  // Model & Dataset info
  const [modelInfo, setModelInfo] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);

  // Settings & Training state
  const [trainingPassword, setTrainingPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingState, setTrainingState] = useState(null);
  const fileInputRef = useRef(null);

  // Scroll handler for navbar
  useEffect(() => {
    const handleScroll = () => {
      setNavScrolled(window.scrollY > 40);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Fetch initial info
  useEffect(() => {
    fetchModelInfo();
    fetchDatasetStats();
  }, []);

  // Poll training status when modal open or training active
  useEffect(() => {
    let interval;
    if (isTraining || showSettingsModal) {
      interval = setInterval(fetchTrainingStatus, 1500);
    }
    return () => clearInterval(interval);
  }, [isTraining, showSettingsModal]);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_BASE}/model/info`);
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.warn('Backend server not reached yet for model info', err);
    }
  };

  const fetchDatasetStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/dataset/stats`);
      if (res.ok) {
        const data = await res.json();
        setDatasetStats(data);
      }
    } catch (err) {
      console.warn('Backend server not reached yet for dataset stats', err);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/train/status`);
      if (res.ok) {
        const data = await res.json();
        setTrainingState(data);
        setIsTraining(data.is_training);
        if (data.status === 'completed') {
          fetchModelInfo();
        }
      }
    } catch (err) {
      console.warn('Could not fetch training status', err);
    }
  };

  // Image Selection Handler
  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setErrorMessage('Please select a valid image file (JPG, PNG, BMP)');
      return;
    }
    setErrorMessage('');
    setAnalysisResult(null);
    setSelectedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  // Image Analysis Call
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage('');
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Failed to analyze image');
      }

      setAnalysisResult(data);
    } catch (err) {
      setErrorMessage(err.message || 'Error connecting to backend API');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Start Training Handler
  const handleStartTraining = async (e) => {
    e.preventDefault();
    setPasswordError('');

    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: trainingPassword }),
      });

      const data = await res.json();

      if (!res.ok) {
        setPasswordError(data.error || 'Authentication failed');
        return;
      }

      setIsTraining(true);
      setTrainingPassword('');
      fetchTrainingStatus();
    } catch (err) {
      setPasswordError('Failed to trigger training process');
    }
  };

  const resetSelection = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setAnalysisResult(null);
    setErrorMessage('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="app">
      {/* NAVBAR */}
      <nav className={`navbar ${navScrolled ? 'scrolled' : ''}`}>
        <div className="container">
          <a href="#" className="nav-logo">
            <div className="logo-icon">🛡️</div>
            <span>DeepGuard <span className="gradient-text">AI</span></span>
          </a>

          <ul className={`nav-links ${mobileMenuOpen ? 'open' : ''}`}>
            <li><a href="#features" onClick={() => setMobileMenuOpen(false)}>Features</a></li>
            <li><a href="#how-it-works" onClick={() => setMobileMenuOpen(false)}>How it Works</a></li>
            <li><a href="#detect" onClick={() => setMobileMenuOpen(false)}>Detector</a></li>
            <li><a href="#architecture" onClick={() => setMobileMenuOpen(false)}>Architecture</a></li>
            <li><a href="#about" onClick={() => setMobileMenuOpen(false)}>About</a></li>
          </ul>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              className="nav-settings-btn"
              onClick={() => setShowSettingsModal(true)}
              title="Settings & Model Training"
            >
              ⚙️
            </button>

            <button
              className="nav-mobile-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? '✕' : '☰'}
            </button>
          </div>
        </div>
      </nav>

      {/* HERO SECTION */}
      <section className="hero">
        <div className="hero-bg">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
          <div className="hero-grid"></div>
        </div>

        <div className="hero-content">
          <div className="hero-badge">
            <span className="dot"></span>
            ResNet-50 Powered Explainable AI
          </div>

          <h1 className="hero-title">
            Detect <span className="highlight">AI-Generated</span> & Fake Images Instantly
          </h1>

          <p className="hero-description">
            Advanced forensic verification using fine-tuned Convolutional Neural Networks,
            Error Level Analysis (ELA), Frequency FFT Spectrum, and Grad-CAM visual heatmaps.
          </p>

          <div className="hero-buttons">
            <a href="#detect" className="btn-primary">
              ⚡ Verify Image Now
            </a>
            <a href="#features" className="btn-secondary">
              🔍 Explore Features
            </a>
          </div>

          <div className="hero-stats-mini">
            <div className="hero-stat-item">
              <div className="value">97.17%</div>
              <div className="label">Validation Accuracy</div>
            </div>
            <div className="hero-stat-item">
              <div className="value">ResNet-50</div>
              <div className="label">Deep Network</div>
            </div>
            <div className="hero-stat-item">
              <div className="value">3-Layer</div>
              <div className="label">XAI Forensics</div>
            </div>
          </div>
        </div>
      </section>

      {/* STATS BAR */}
      <section className="stats-bar">
        <div className="container">
          <div className="stat-card">
            <div className="stat-icon">🎯</div>
            <div className="stat-value">97.17%</div>
            <div className="stat-label">Model Accuracy</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">⚡</div>
            <div className="stat-value">&lt; 0.5s</div>
            <div className="stat-label">Inference Speed</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🔬</div>
            <div className="stat-value">3 Modes</div>
            <div className="stat-label">Explainable AI</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🖼️</div>
            <div className="stat-value">
              {datasetStats ? datasetStats.total : '2,000+'}
            </div>
            <div className="stat-label">Trained Dataset Images</div>
          </div>
        </div>
      </section>

      {/* FEATURES SHOWCASE */}
      <section id="features" className="section">
        <div className="container">
          <h2 className="section-title">Forensic Feature Suite</h2>
          <p className="section-subtitle">
            Combines deep classification with traditional image forensics and explainable visual attention maps.
          </p>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🧠</div>
              <h3>ResNet-50 Transfer Learning</h3>
              <p>
                Fine-tuned 50-layer deep neural network trained specifically on real photography vs synthetic AI diffusion renders.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h3>Error Level Analysis (ELA)</h3>
              <p>
                Exposes compression rate variations across image regions, highlighting digital splices and synthetic resaving.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🌐</div>
              <h3>Frequency Domain FFT</h3>
              <p>
                Analyzes 2D Fourier transforms to detect unnatural periodic artifact patterns typical of generative AI upscalers.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">🔥</div>
              <h3>Grad-CAM Heatmaps</h3>
              <p>
                Gradient-weighted Class Activation Mapping displays exact pixel regions that guided the AI's prediction.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-Time Web API</h3>
              <p>
                Asynchronous backend processing engine delivering complete visual breakdown in under half a second.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-icon">⚙️</div>
              <h3>On-Demand Model Training</h3>
              <p>
                Integrated admin portal allowing model fine-tuning with full dataset augmentation directly from settings.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how-it-works" className="section how-it-works">
        <div className="container">
          <h2 className="section-title">How DeepGuard Works</h2>
          <p className="section-subtitle">
            Simple 3-step verification workflow for high-precision authenticity detection.
          </p>

          <div className="steps-container">
            <div className="step-card">
              <div className="step-number">
                <span className="icon">📤</span>
              </div>
              <h3>1. Upload Image</h3>
              <p>Drag and drop or select any photograph (JPG, PNG, WEBP) from your device.</p>
            </div>

            <div className="step-card">
              <div className="step-number">
                <span className="icon">⚙️</span>
              </div>
              <h3>2. Deep Forensic Pipeline</h3>
              <p>ResNet-50 computes probabilities while ELA, FFT, and Grad-CAM extract spatial features.</p>
            </div>

            <div className="step-card">
              <div className="step-number">
                <span className="icon">📊</span>
              </div>
              <h3>3. Visual Verdict Report</h3>
              <p>Receive immediate Real/Fake verdict with confidence score and side-by-side visual evidence.</p>
            </div>
          </div>
        </div>
      </section>

      {/* DETECTOR SECTION */}
      <section id="detect" className="section detection">
        <div className="container">
          <h2 className="section-title">Interactive Detector</h2>
          <p className="section-subtitle">
            Test any image to classify whether it is a authentic photograph or AI-generated artifact.
          </p>

          <div className="detection-wrapper">
            {!imagePreview ? (
              <div
                className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                />
                <div className="upload-icon">📸</div>
                <h3>Drop your image here, or click to browse</h3>
                <p>Supports JPG, PNG, WEBP up to 20MB</p>
                <div className="browse-btn">Choose Image</div>
              </div>
            ) : (
              <div className="image-preview-container">
                <div className="preview-header">
                  <h4>📷 Selected Image</h4>
                  <button className="remove-btn" onClick={resetSelection}>
                    ✕ Remove & Upload Another
                  </button>
                </div>

                <div className="preview-image">
                  <img src={imagePreview} alt="Preview" />
                </div>

                {!analysisResult && !isAnalyzing && (
                  <div className="analyze-button-container">
                    <button className="btn-primary analyze-btn" onClick={handleAnalyze}>
                      🚀 Run Detection & Forensic Analysis
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* ERROR DISPLAY */}
            {errorMessage && (
              <div className="error-message" style={{ marginTop: '20px' }}>
                ⚠️ {errorMessage}
              </div>
            )}

            {/* LOADING SPINNER */}
            {isAnalyzing && (
              <div className="loading-overlay">
                <div className="spinner"></div>
                <div className="loading-text">
                  Analyzing spatial features, calculating ELA compression differences & Grad-CAM map...
                </div>
              </div>
            )}

            {/* RESULTS REPORT */}
            {analysisResult && (
              <div className="results-container animate-in">
                <div className={`verdict-card ${analysisResult.prediction.is_fake ? 'fake' : 'real'}`}>
                  <div className={`verdict-header ${analysisResult.prediction.is_fake ? 'fake' : 'real'}`}>
                    <div className={`verdict-icon ${analysisResult.prediction.is_fake ? 'fake' : 'real'}`}>
                      {analysisResult.prediction.is_fake ? '🚨' : '✅'}
                    </div>

                    <div className="verdict-info">
                      <h2 className={analysisResult.prediction.is_fake ? 'fake' : 'real'}>
                        {analysisResult.prediction.prediction === 'FAKE'
                          ? 'FAKE (AI-Generated)'
                          : 'REAL (Original Photograph)'}
                      </h2>
                      <div className="confidence-text">
                        Detection Confidence: <strong>{analysisResult.prediction.confidence}%</strong>
                      </div>
                    </div>

                    <div className="confidence-bar-wrapper">
                      <div className="confidence-bar-bg">
                        <div
                          className={`confidence-bar-fill ${analysisResult.prediction.is_fake ? 'fake' : 'real'}`}
                          style={{ width: `${analysisResult.prediction.confidence}%` }}
                        ></div>
                      </div>
                      <div className="confidence-labels">
                        <span>Real: {analysisResult.prediction.scores.REAL?.toFixed(1)}%</span>
                        <span>Fake: {analysisResult.prediction.scores.FAKE?.toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* XAI VISUAL ANALYSIS TRIPLE CARD */}
                <h3 style={{ marginBottom: '20px', fontSize: '1.2rem', textAlign: 'center' }}>
                  🔬 Explainable AI Visual Analysis
                </h3>

                <div className="analysis-grid">
                  {/* ELA */}
                  <div className="analysis-card">
                    <div className="card-header">
                      <span>📉</span> Error Level Analysis (ELA)
                    </div>
                    <div className="card-image">
                      {analysisResult.analysis.ela?.image ? (
                        <img
                          src={`data:image/png;base64,${analysisResult.analysis.ela.image}`}
                          alt="ELA Analysis"
                        />
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>ELA Unavailable</span>
                      )}
                    </div>
                    <div className="card-description">
                      {analysisResult.analysis.ela?.description ||
                        'Highlights compression differences across regions.'}
                    </div>
                  </div>

                  {/* FFT */}
                  <div className="analysis-card">
                    <div className="card-header">
                      <span>🌐</span> Frequency Spectrum (FFT)
                    </div>
                    <div className="card-image">
                      {analysisResult.analysis.fft?.image ? (
                        <img
                          src={`data:image/png;base64,${analysisResult.analysis.fft.image}`}
                          alt="FFT Spectrum"
                        />
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>FFT Unavailable</span>
                      )}
                    </div>
                    <div className="card-description">
                      {analysisResult.analysis.fft?.description ||
                        'Exposes frequency periodicities typical in synthetic images.'}
                    </div>
                  </div>

                  {/* GRAD-CAM */}
                  <div className="analysis-card">
                    <div className="card-header">
                      <span>🔥</span> Grad-CAM Attention Map
                    </div>
                    <div className="card-image">
                      {analysisResult.analysis.gradcam?.image ? (
                        <img
                          src={`data:image/png;base64,${analysisResult.analysis.gradcam.image}`}
                          alt="Grad-CAM"
                        />
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>
                          {analysisResult.analysis.gradcam?.error || 'Grad-CAM Unavailable'}
                        </span>
                      )}
                    </div>
                    <div className="card-description">
                      {analysisResult.analysis.gradcam?.description ||
                        'Red regions highlight key pixels influencing decision.'}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ARCHITECTURE SECTION */}
      <section id="architecture" className="section architecture">
        <div className="container">
          <h2 className="section-title">Network Architecture</h2>
          <p className="section-subtitle">
            Transfer learning configuration based on ResNet-50 with custom classification head.
          </p>

          <div className="arch-pipeline">
            <div className="arch-step">
              <div className="step-label">Input</div>
              <div className="step-name">224 × 224 RGB</div>
            </div>

            <div className="arch-arrow">➔</div>

            <div className="arch-step">
              <div className="step-label">Feature Extractor</div>
              <div className="step-name">ResNet-50 Layers</div>
            </div>

            <div className="arch-arrow">➔</div>

            <div className="arch-step">
              <div className="step-label">Activation Hook</div>
              <div className="step-name">activation_49_relu</div>
            </div>

            <div className="arch-arrow">➔</div>

            <div className="arch-step">
              <div className="step-label">Classifier</div>
              <div className="step-name">FC (2 Classes)</div>
            </div>

            <div className="arch-arrow">➔</div>

            <div className="arch-step">
              <div className="step-label">Output</div>
              <div className="step-name">REAL / FAKE</div>
            </div>
          </div>

          <div className="arch-details">
            <div className="arch-detail-card">
              <h3>⚙️ Training Hyperparameters</h3>
              <ul>
                <li>
                  <span className="detail-label">Base Architecture</span>
                  <span className="detail-value">ResNet-50</span>
                </li>
                <li>
                  <span className="detail-label">Optimizer</span>
                  <span className="detail-value">SGDM (Momentum 0.9)</span>
                </li>
                <li>
                  <span className="detail-label">Initial Learn Rate</span>
                  <span className="detail-value">0.001</span>
                </li>
                <li>
                  <span className="detail-label">Learning Schedule</span>
                  <span className="detail-value">Piecewise Drop (0.1× / 2 Epochs)</span>
                </li>
                <li>
                  <span className="detail-label">Data Augmentation</span>
                  <span className="detail-value">Random Flip & Rotation (±20°)</span>
                </li>
              </ul>
            </div>

            <div className="arch-detail-card">
              <h3>📊 Evaluation Metrics</h3>
              <ul>
                <li>
                  <span className="detail-label">Validation Accuracy</span>
                  <span className="detail-value">97.17%</span>
                </li>
                <li>
                  <span className="detail-label">Dataset Split</span>
                  <span className="detail-value">70% Train / 15% Val / 15% Test</span>
                </li>
                <li>
                  <span className="detail-label">Epochs</span>
                  <span className="detail-value">4 Epochs with Checkpoints</span>
                </li>
                <li>
                  <span className="detail-label">Mini Batch Size</span>
                  <span className="detail-value">8 Samples</span>
                </li>
                <li>
                  <span className="detail-label">Checkpoints Saved</span>
                  <span className="detail-value">Checkpoints_Fresh/*.mat</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ABOUT SECTION */}
      <section id="about" className="section">
        <div className="container">
          <div className="about-card">
            <div className="about-avatar">S</div>
            <h3>Suhas S Kattimani</h3>
            <div className="about-role">B.Tech – Information Science and Engineering</div>
            <div className="about-university">Presidency University</div>

            <p>
              Developer and creator of the AI-Powered Fake Image Detection project.
              Engineered using PyTorch/MATLAB Deep Learning Toolboxes, transfer learning on ResNet-50,
              and Explainable AI modules for transparent media authenticity verification.
            </p>

            <div className="about-links">
              <a
                href="https://github.com/SuhasReturn/Fake-Image-Detection"
                target="_blank"
                rel="noreferrer"
                className="about-link"
              >
                <span>⭐</span> GitHub Repository
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* SETTINGS & TRAINING MODAL */}
      {showSettingsModal && (
        <div className="modal-overlay" onClick={() => setShowSettingsModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>⚙️ System Settings & Model Training</h2>
              <button className="modal-close" onClick={() => setShowSettingsModal(false)}>
                ✕
              </button>
            </div>

            <div className="modal-body">
              {/* CURRENT MODEL STATUS */}
              <div className="settings-section">
                <h3>ℹ️ Loaded Model Status</h3>
                <div className="model-info-grid">
                  <div className="model-info-item">
                    <div className="info-label">Architecture</div>
                    <div className="info-value">{modelInfo?.architecture || 'ResNet-50'}</div>
                  </div>
                  <div className="model-info-item">
                    <div className="info-label">Accuracy</div>
                    <div className="info-value">
                      {modelInfo?.accuracy ? `${modelInfo.accuracy}%` : '97.17%'}
                    </div>
                  </div>
                  <div className="model-info-item">
                    <div className="info-label">Input Resolution</div>
                    <div className="info-value">{modelInfo?.input_size || '224x224'}</div>
                  </div>
                  <div className="model-info-item">
                    <div className="info-label">Status</div>
                    <div className="info-value" style={{ color: 'var(--accent-emerald)' }}>
                      Ready
                    </div>
                  </div>
                </div>
              </div>

              {/* TRAIN MODEL FORM */}
              <div className="settings-section">
                <h3>🏋️ Train / Fine-Tune Model</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Restricted action. Enter admin password to execute model training on dataset.
                </p>

                <form onSubmit={handleStartTraining}>
                  <div className="settings-field">
                    <label>Admin Password</label>
                    <input
                      type="password"
                      placeholder="Enter training password..."
                      value={trainingPassword}
                      onChange={(e) => setTrainingPassword(e.target.value)}
                      disabled={isTraining}
                    />
                  </div>

                  {passwordError && <div className="error-message">{passwordError}</div>}

                  <button
                    type="submit"
                    className="btn-primary"
                    style={{ width: '100%', marginTop: '12px', justifyContent: 'center' }}
                    disabled={isTraining || !trainingPassword}
                  >
                    {isTraining ? '⏳ Training in Progress...' : '🚀 Start 4-Epoch Model Training'}
                  </button>
                </form>
              </div>

              {/* LIVE TRAINING PROGRESS & LOG */}
              {trainingState && (trainingState.is_training || trainingState.progress > 0) && (
                <div className="settings-section training-progress">
                  <h3>📈 Live Training Progress</h3>

                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${trainingState.progress}%` }}
                    ></div>
                  </div>

                  <div className="training-stats">
                    <div className="training-stat">
                      <div className="stat-name">Epoch</div>
                      <div className="stat-val">
                        {trainingState.epoch} / {trainingState.total_epochs}
                      </div>
                    </div>
                    <div className="training-stat">
                      <div className="stat-name">Loss</div>
                      <div className="stat-val">{trainingState.loss}</div>
                    </div>
                    <div className="training-stat">
                      <div className="stat-name">Val Acc</div>
                      <div className="stat-val">{trainingState.val_accuracy}%</div>
                    </div>
                  </div>

                  <div className="training-log">
                    {trainingState.log && trainingState.log.length > 0 ? (
                      trainingState.log.map((line, idx) => (
                        <div key={idx} className="log-entry">
                          {line}
                        </div>
                      ))
                    ) : (
                      <div className="log-entry">Initializing training thread...</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* FOOTER */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="logo-icon">🛡️</div>
              <span>DeepGuard AI</span>
            </div>

            <div className="footer-links">
              <a href="#features">Features</a>
              <a href="#how-it-works">Workflow</a>
              <a href="#detect">Detector</a>
              <a href="https://github.com/SuhasReturn/Fake-Image-Detection" target="_blank" rel="noreferrer">
                GitHub
              </a>
            </div>
          </div>

          <div className="footer-bottom">
            © {new Date().getFullYear()} Suhas S Kattimani | Fake Image Detection System | Built with React & PyTorch
          </div>
        </div>
      </footer>
    </div>
  );
}
