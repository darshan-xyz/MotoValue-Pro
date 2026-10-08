let isSubmitting = false;

function showError(message) {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    if (errorAlert && errorMessage) {
        errorMessage.innerText = message;
        errorAlert.classList.remove('hidden');
        errorAlert.style.display = 'block';
    }
}

function clearError() {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    if (errorAlert && errorMessage) {
        errorMessage.innerText = '';
        errorAlert.classList.add('hidden');
        errorAlert.style.display = 'none';
    }
}

function setLoading(isLoading) {
    const submitBtn = document.getElementById('submitBtn');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');

    isSubmitting = isLoading;
    if (submitBtn) {
        submitBtn.disabled = isLoading;
    }
    if (btnSpinner) {
        if (isLoading) {
            btnSpinner.classList.remove('hidden');
            btnSpinner.style.display = 'inline-block';
        } else {
            btnSpinner.classList.add('hidden');
            btnSpinner.style.display = 'none';
        }
    }
    if (btnText) {
        btnText.innerText = isLoading ? "Calculating..." : "Calculate Value";
    }
}

async function predictPrice() {
    if (isSubmitting) {
        return;
    }

    clearError();

    // 1. Collect Data from the HTML Form
    const data = {
        brand: document.getElementById('brand').value,
        model: document.getElementById('model').value,
        year: document.getElementById('year').value,
        km_driven: document.getElementById('km_driven').value,
        mileage: document.getElementById('mileage').value,
        engine: document.getElementById('engine').value,
        max_power: document.getElementById('max_power').value,
        seats: document.getElementById('seats').value,
        fuel: document.getElementById('fuel').value,
        transmission: document.getElementById('transmission').value,
        seller_type: document.getElementById('seller_type').value,
        owner: document.getElementById('owner').value
    };

    // Client-side validation to prevent sending empty brand/model
    if (!data.brand || !data.model) {
        showError("Please select a Brand and Model first.");
        return;
    }

    setLoading(true);

    // 2. Send Data to Python Backend via Fetch API
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // 3. Update the UI with the result
        if (response.ok && result.price) {
            const priceText = document.getElementById('priceText');
            const resultCard = document.getElementById('result');

            // Set the price
            priceText.innerText = "₹ " + result.price;

            // Make the result card visible
            resultCard.classList.remove('hidden');
            resultCard.style.visibility = 'visible';
            resultCard.style.display = 'block';

            // Smooth scroll to the result
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            let errorMsg = "Prediction failed.";
            if (result.details && Array.isArray(result.details)) {
                errorMsg = result.details.map(d => `${d.field}: ${d.message}`).join(", ");
            } else if (result.error) {
                errorMsg = result.error + (result.message ? `: ${result.message}` : "");
            }
            showError(errorMsg);
        }
    } catch (error) {
        console.error("Prediction Request Error:", error);
        showError("Unable to reach the prediction service. Please check your connection and try again.");
    } finally {
        setLoading(false);
    }
}