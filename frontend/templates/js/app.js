// document.addEventListener("DOMContentLoaded", () => {
//     const form = document.getElementById("churn-form");

//     form.addEventListener("submit", (event) => {
//         event.preventDefault(); // Prevent form submission

//         // Collect form data
//         const formData = new FormData(form);
//         const customerData = Object.fromEntries(formData.entries());

//         // Process and predict churn (placeholder logic)
//         const churnProbability = calculateChurnProbability(customerData);
//         const churnGroup = churnProbability > 0.5 ? "High Risk" : "Low Risk";

//         // Update outputs
//         document.getElementById("predicted-churn-probability").textContent = `${(churnProbability * 100).toFixed(2)}%`;
//         document.getElementById("churn-group").textContent = churnGroup;
//     });

//     /**
//      * Placeholder function to calculate churn probability
//      * Replace with real prediction logic or API integration
//      */
//     function calculateChurnProbability(data) {
//         // Basic scoring logic for demo purposes
//         const ageFactor = data.age ? parseInt(data.age, 10) / 100 : 0;
//         const loyaltyFactor = data["policy-loyalty-score"] ? parseInt(data["policy-loyalty-score"], 10) / 100 : 0;

//         return Math.min(ageFactor + loyaltyFactor, 1); // Returns a probability between 0 and 1
//     }
// });


document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("predict-btn").addEventListener("click", function (event) {
        event.preventDefault();

        // Collect form data
        const formData = new FormData(document.getElementById("churn-form"));
        let jsonData = {};
        formData.forEach((value, key) => {
            jsonData[key] = value;
        });

        // Send data to Flask API
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(jsonData),
        })
            .then((response) => response.json())
            .then((data) => {
                // Update output fields with the prediction results
                document.getElementById("predicted-churn-probability").textContent = data.churn_probability;
                document.getElementById("churn-group").textContent = data.churn_group;
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    });
});
