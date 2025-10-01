document.getElementById("predict-btn").addEventListener("click", async function (event) {
  event.preventDefault();
  const form = document.getElementById("churn-form");
  const formData = Object.fromEntries(new FormData(form).entries());

  // Validate required fields
  const requiredFields = ["Age", "Policy_Tenure", "Policy_Renewal_Count", "Policy_Cancellation_History", "Premium_Amount", "Total_Claims_Filed", "Claim_Amount_Total", "Family_Members_Insured", "Missed_Payments", "Customer_Complaints", "Complaint_Resolution_Time", "Customer_Support_Calls", "Vehicle_Age"];
  for (let key of requiredFields) {
    const value = formData[key];
    if (!value || parseFloat(value) < 0) {
      alert(`Invalid input for ${key}`);
      return;
    }
  }

  // Predict basic churn
  const predictResponse = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  });
  const predictResult = await predictResponse.json();

  if (predictResult.error) return alert("Error: " + predictResult.error);

  // Populate prediction summary
  document.getElementById("predicted-churn-probability").innerText = predictResult.probability + "%";
  document.getElementById("churn-group").innerText = predictResult.prediction;

  // Show output section
  document.getElementById("output-section").style.display = "block";

  const churnReasonsBtn = document.getElementById("churn-reasons-btn");
  const retentionStrategyBtn = document.getElementById("retention-strategy-btn");

  if (predictResult.prediction === "Churn") {
    churnReasonsBtn.disabled = false;
    retentionStrategyBtn.disabled = false;
  } else {
    churnReasonsBtn.disabled = true;
    retentionStrategyBtn.disabled = true;
    document.getElementById("churn-reasons-text").innerText = "No churn predicted.";
    document.getElementById("retention-strategy-text").innerText = "Retention strategy not needed.";
  }
});

// Churn Reason Button
document.getElementById("churn-reasons-btn").addEventListener("click", async () => {
  const formData = Object.fromEntries(new FormData(document.getElementById("churn-form")).entries());

  const explainResponse = await fetch('/predict_and_explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  });
  const explainResult = await explainResponse.json();

  if (explainResult.error) return alert("Error: " + explainResult.error);

  document.getElementById("churn-reasons-text").innerText = explainResult.explanation;

  if (explainResult.shap_plot_url) {
    const shapImg = document.getElementById("shap-plot");
    shapImg.src = explainResult.shap_plot_url;
    shapImg.style.display = "block";
  }
});

// Retention Strategy Button
document.getElementById("retention-strategy-btn").addEventListener("click", async () => {
  const formData = Object.fromEntries(new FormData(document.getElementById("churn-form")).entries());

  const response = await fetch('/generate_retention_strategy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(formData)
  });
  const result = await response.json();

  if (result.error) return alert("Error: " + result.error);

  document.getElementById("retention-strategy-text").innerText = result.strategies;
});

// AI Explanation Buttons
document.getElementById("explain-churn-reason-ai").addEventListener("click", async () => {
  const topFeaturesText = document.getElementById("churn-reasons-text").innerText;
  const formElements = document.getElementById("churn-form").elements;
  const rawData = {};
  for (let elem of formElements) {
    if (elem.name) rawData[elem.name] = elem.value;
  }

  const response = await fetch('/explain_churn_reason_ai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ top_features_text: topFeaturesText, raw_data: rawData })
  });
  const result = await response.json();
  document.getElementById("ai-churn-reason-explanation").innerText = result.explanation || "No explanation available.";
});

document.getElementById("explain-retention-strategy-ai").addEventListener("click", async () => {
  const rawData = Object.fromEntries(new FormData(document.getElementById("churn-form")).entries());

  const response = await fetch('/explain_retention_strategy_ai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_data: rawData })  // strategy_text not needed anymore
  });
  const result = await response.json();
  document.getElementById("ai-retention-strategy-explanation").innerText = result.explanation || "No explanation available.";
});
