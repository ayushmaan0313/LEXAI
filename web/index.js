/**
 * LEXAI Dashboard — Frontend Logic
 * Handles: drag-and-drop upload, API calls, dynamic results rendering
 */

(function () {
    "use strict";

    // --- DOM Elements ---
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadSection = document.getElementById("uploadSection");
    const uploadPreview = document.getElementById("uploadPreview");
    const previewImage = document.getElementById("previewImage");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resetBtn = document.getElementById("resetBtn");
    const loadingSection = document.getElementById("loadingSection");
    const loadingStep = document.getElementById("loadingStep");
    const loadingBarFill = document.getElementById("loadingBarFill");
    const resultsSection = document.getElementById("resultsSection");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    let selectedFile = null;
    const API_BASE = window.location.origin;

    // --- Initialization ---
    checkHealth();

    // --- Health Check ---
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();
            statusDot.classList.add("online");
            statusDot.classList.remove("offline");
            statusText.textContent = data.model_loaded
                ? `Ready (${data.device})`
                : `No model (${data.device})`;
        } catch {
            statusDot.classList.add("offline");
            statusDot.classList.remove("online");
            statusText.textContent = "API Offline";
        }
    }

    // --- Upload Handlers ---
    uploadZone.addEventListener("click", () => fileInput.click());

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload an image file.");
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZone.style.display = "none";
            uploadPreview.style.display = "block";
            resultsSection.style.display = "none";
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        uploadZone.style.display = "";
        uploadPreview.style.display = "none";
        resultsSection.style.display = "none";
        loadingSection.style.display = "none";
    });

    // --- Analyze ---
    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        // Show loading
        uploadSection.style.display = "none";
        loadingSection.style.display = "";
        resultsSection.style.display = "none";

        // Simulate progress
        const steps = [
            "Processing image through CNN ensemble...",
            "Running cell segmentation...",
            "Constructing cell graph...",
            "GNN spatial analysis...",
            "Multi-modal fusion...",
            "Generating Grad-CAM heatmaps...",
            "Estimating uncertainty (MC Dropout)...",
            "Compiling results...",
        ];

        let stepIdx = 0;
        const stepInterval = setInterval(() => {
            if (stepIdx < steps.length) {
                loadingStep.textContent = steps[stepIdx];
                loadingBarFill.style.width = `${((stepIdx + 1) / steps.length) * 90}%`;
                stepIdx++;
            }
        }, 800);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const response = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                body: formData,
            });

            clearInterval(stepInterval);
            loadingBarFill.style.width = "100%";
            loadingStep.textContent = "Analysis complete!";

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Analysis failed");
            }

            const result = await response.json();

            setTimeout(() => {
                loadingSection.style.display = "none";
                uploadSection.style.display = "";
                renderResults(result);
            }, 500);
        } catch (error) {
            clearInterval(stepInterval);
            loadingSection.style.display = "none";
            uploadSection.style.display = "";
            alert(`Analysis Error: ${error.message}`);
        }
    });

    // --- Render Results ---
    // Friendly display names for cancer subtypes
    const DISPLAY_NAMES = {
        "ALL_Blast": "ALL (Blast)",
        "ALL_Early_Pre_B": "ALL — Early Pre-B",
        "ALL_Pre_B": "ALL — Pre-B",
        "ALL_Pro_B": "ALL — Pro-B",
        "Benign": "Benign (Normal)",
    };

    function friendlyName(cls) {
        return DISPLAY_NAMES[cls] || cls;
    }

    function renderResults(data) {
        resultsSection.style.display = "";

        // Classification
        document.getElementById("predictedClass").textContent = friendlyName(data.predicted_class);

        // Confidence
        const confPct = (data.confidence * 100).toFixed(1);
        document.getElementById("confidenceValue").textContent = `${confPct}%`;

        // Uncertainty badge
        const badge = document.getElementById("uncertaintyBadge");
        const badgeText = document.getElementById("uncertaintyText");
        if (data.is_uncertain) {
            badge.className = "uncertainty-badge uncertain";
            badgeText.textContent = "⚠ Uncertain — Needs Review";
        } else {
            badge.className = "uncertainty-badge confident";
            badgeText.textContent = "✓ Confident";
        }

        // Probability bars
        renderProbabilityBars(data.probabilities);

        // Grad-CAM
        const gradcamImg = document.getElementById("gradcamImage");
        if (data.gradcam_heatmap) {
            gradcamImg.src = data.gradcam_heatmap;
            gradcamImg.style.display = "";
        } else {
            gradcamImg.src = "";
            gradcamImg.style.display = "none";
        }

        // GNN Graph
        const gnnImg = document.getElementById("gnnGraphImage");
        if (data.gnn_graph_visualization) {
            gnnImg.src = data.gnn_graph_visualization;
            gnnImg.style.display = "";
        } else {
            gnnImg.src = "";
            gnnImg.style.display = "none";
        }

        // Cell Metrics
        document.getElementById("cellCount").textContent = data.cell_count;
        document.getElementById("blastPercentage").textContent =
            `${data.blast_percentage.toFixed(1)}%`;
        document.getElementById("spatialScore").textContent =
            data.spatial_pattern_score.toFixed(3);
        document.getElementById("modelConfidence").textContent =
            `${confPct}%`;

        // Backbone weights
        renderBackboneBars(data.cnn_backbone_weights);

        // Uncertainty details
        renderUncertaintyDetails(data);

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function renderProbabilityBars(probs) {
        const container = document.getElementById("probabilityBars");
        container.innerHTML = "";

        const classColors = {
            ALL_Blast: "all",
            ALL_Early_Pre_B: "all",
            ALL_Pre_B: "aml",
            ALL_Pro_B: "cml",
            Benign: "normal",
        };

        const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        for (const [cls, prob] of sortedEntries) {
            const pct = (prob * 100).toFixed(1);
            const colorClass = classColors[cls] || "all";

            const item = document.createElement("div");
            item.className = "prob-bar-item";
            item.innerHTML = `
                <span class="prob-label">${friendlyName(cls)}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${colorClass}" style="width: 0%"></div>
                </div>
                <span class="prob-value">${pct}%</span>
            `;
            container.appendChild(item);

            // Animate
            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector(".prob-bar-fill").style.width = `${pct}%`;
                }, 100);
            });
        }
    }

    function renderBackboneBars(weights) {
        const container = document.getElementById("backboneBars");
        container.innerHTML = "";

        const friendlyNames = {
            efficientnet: "EfficientNet",
            resnet50: "ResNet50",
            densenet121: "DenseNet121",
        };

        if (!weights || Object.keys(weights).length === 0) {
            container.innerHTML =
                '<p style="color:var(--text-muted);text-align:center;font-size:0.85rem;">No backbone weight data</p>';
            return;
        }

        for (const [name, weight] of Object.entries(weights)) {
            const pct = (weight * 100).toFixed(1);
            const displayName = friendlyNames[name] || name;

            const item = document.createElement("div");
            item.className = "backbone-item";
            item.innerHTML = `
                <span class="backbone-name">${displayName}</span>
                <div class="backbone-bar-track">
                    <div class="backbone-bar-fill" style="width: 0%"></div>
                </div>
                <span class="backbone-weight">${pct}%</span>
            `;
            container.appendChild(item);

            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector(".backbone-bar-fill").style.width = `${pct}%`;
                }, 150);
            });
        }
    }

    function renderUncertaintyDetails(data) {
        const container = document.getElementById("uncertaintyDetails");
        container.innerHTML = "";

        const classes = Object.keys(data.probabilities || {});
        const probs = data.probabilities || {};
        const ciLow = data.confidence_interval_low || {};
        const ciHigh = data.confidence_interval_high || {};
        const variance = data.prediction_variance || {};

        for (const cls of classes) {
            const mean = probs[cls] || 0;
            const low = ciLow[cls] || 0;
            const high = ciHigh[cls] || 0;

            const row = document.createElement("div");
            row.className = "uncertainty-row";
            row.innerHTML = `
                <span class="uncertainty-class">${friendlyName(cls)}</span>
                <div class="uncertainty-bar-container">
                    <div class="uncertainty-bar-bg"></div>
                    <div class="uncertainty-ci" style="left:${low * 100}%;width:${(high - low) * 100}%"></div>
                    <div class="uncertainty-mean" style="left:calc(${mean * 100}% - 2px)"></div>
                </div>
            `;
            container.appendChild(row);
        }

        // Add overall confidence summary
        const summary = document.createElement("div");
        summary.style.cssText =
            "margin-top:12px;padding:12px 16px;background:var(--bg-glass);border-radius:var(--radius-sm);border:1px solid var(--border-glass);text-align:center;";

        const confLevel = data.confidence >= 0.9 ? "High" : data.confidence >= 0.7 ? "Moderate" : "Low";
        const confColor = data.confidence >= 0.9 ? "var(--success)" : data.confidence >= 0.7 ? "var(--warning)" : "var(--danger)";

        summary.innerHTML = `
            <span style="font-size:0.8rem;color:var(--text-muted);">Overall Confidence Level: </span>
            <span style="font-size:0.9rem;font-weight:700;color:${confColor};font-family:var(--font-mono);">${confLevel} (${(data.confidence * 100).toFixed(1)}%)</span>
        `;
        container.appendChild(summary);
    }
})();
