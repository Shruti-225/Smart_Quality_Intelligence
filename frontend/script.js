async function loadDashboard() {

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/api/dashboard"
        );

        const data = await response.json();

        document.getElementById("totalProduction").textContent =
            data.total_production;

        document.getElementById("goodUnits").textContent =
            data.good_units;

        document.getElementById("defectiveUnits").textContent =
            data.defective_units;

        document.getElementById("defectRate").textContent =
            data.defect_rate + "%";

        document.getElementById("qualityLoss").textContent =
            "₹" + data.quality_loss.toLocaleString("en-IN");

        document.getElementById("potentialSavings").textContent =
            "₹" + data.potential_savings.toLocaleString("en-IN");

        document.getElementById("bestModel").textContent =
            data.best_model;

        document.getElementById("rocAuc").textContent =
            data.best_model_roc_auc;

        document.getElementById("f1Score").textContent =
            data.best_model_f1;

        document.getElementById("visionDetection").textContent =
            data.vision_detection_rate + "%";

        document.getElementById("difficultDefect").textContent =
            data.difficult_defect;

        document.getElementById("difficultDefectRate").textContent =
            data.difficult_defect_rate + "%";   

        document.getElementById("status").textContent =
            "✓ Backend connected successfully";

    }

    catch (error) {

        console.error(error);

        document.getElementById("status").textContent =
            "✗ Could not connect to backend. Please start Flask.";

    }
}

loadDashboard();