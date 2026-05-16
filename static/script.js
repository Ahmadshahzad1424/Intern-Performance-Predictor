document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');
    
    // UI Loading State
    submitBtn.innerText = "Analyzing Patterns...";
    submitBtn.disabled = true;

    const data = {
        task_completion_time: parseFloat(document.getElementById('time').value),
        feedback_rating: parseFloat(document.getElementById('feedback').value),
        attendance_rate: parseFloat(document.getElementById('attendance').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) throw new Error('API Error');

        const result = await response.json();

        // Update UI with results
        resultsContainer.style.display = 'block';
        document.getElementById('predicted-score').innerText = result.score;
        document.getElementById('status-val').innerText = result.status;
        document.getElementById('efficiency-val').innerText = result.metrics.efficiency + "%";
        document.getElementById('recommendation-text').innerText = result.recommendation;

        // Smooth scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error:', error);
        alert('Engine Error: Could not connect to AI Backend');
    } finally {
        submitBtn.innerText = "Run AI Analysis";
        submitBtn.disabled = false;
    }
});
