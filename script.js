const diabetesDataUrl = 'https://raw.githubusercontent.com/Chando0185/Multiple_Disease_Prediction/6807663d7f8c63e5998746f86651db5dadbda674/dataset/diabetes.csv';
const heartDataUrl = 'https://raw.githubusercontent.com/Chando0185/Multiple_Disease_Prediction/6807663d7f8c63e5998746f86651db5dadbda674/dataset/heart.csv';

function fetchData(url, callback) {
    fetch(url)
    .then(response => response.text())
    .then(csvText => {
        Papa.parse(csvText, {
            header: true,
            complete: (results) => {
                callback(results.data);
            }
        });
    })
    .catch(error => console.error("Error loading data:", error));
}

function updateCharts(diabetesData, heartData) {
 
    const positiveDiabetes = diabetesData.filter(item => item.Outcome == "1").length;
    const negativeDiabetes = diabetesData.filter(item => item.Outcome == "0").length;

    const positiveHeart = heartData.filter(item => item.target == "1").length;
    const negativeHeart = heartData.filter(item => item.target == "0").length;


    document.querySelector('.stats-cards .card:nth-child(1) h1').textContent = positiveDiabetes + positiveHeart + negativeDiabetes + negativeHeart;
    document.querySelector('.stats-cards .card:nth-child(3) h1').textContent = 2; 

    // Disease Types (Pie Chart)
    diseaseChart.data.datasets[0].data = [positiveDiabetes, positiveHeart];
    diseaseChart.update();

    // Prediction for Breakdown (Line Chart)
    breakdownChart.data.datasets[0].data = [positiveDiabetes, positiveHeart];
    breakdownChart.update();

    // Predictions for Bar Chart
    predictionsChart.data.datasets[0].data = [positiveDiabetes, positiveHeart]; 
    predictionsChart.data.datasets[1].data = [negativeDiabetes, negativeHeart]; 
    predictionsChart.update();
}

fetchData(diabetesDataUrl, (diabetesData) => {
    fetchData(heartDataUrl, (heartData) => {
        updateCharts(diabetesData, heartData);
    });
});

// Bar Chart
const predictionsCtx = document.getElementById('predictionsChart').getContext('2d');
const predictionsChart = new Chart(predictionsCtx, {
    type: 'bar',
    data: {
        labels: ['Diabetes', 'Heart Disease'],
        datasets: [
            {
                label: 'Positive Predictions',
                data: [0, 0],
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: 'Negative Predictions',
                data: [0, 0], 
                backgroundColor: 'rgba(255, 159, 64, 0.6)',
                borderColor: 'rgba(255, 159, 64, 1)',
                borderWidth: 1
            }
        ]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Pie (Doughnut) Chart
const diseaseCtx = document.getElementById('diseaseChart').getContext('2d');
const diseaseChart = new Chart(diseaseCtx, {
    type: 'doughnut',
    data: {
        labels: ['Diabetes', 'Heart Disease'],
        datasets: [{
            label: 'Top Diseases Predicted',
            data: [0, 0], 
            backgroundColor: [
                'rgba(255, 99, 132, 0.6)',
                'rgba(54, 162, 235, 0.6)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
            ],
            borderWidth: 1
        }]
    }
});

// Line Chart
const breakdownCtx = document.getElementById('breakdownChart').getContext('2d');
const breakdownChart = new Chart(breakdownCtx, {
    type: 'line',
    data: {
        labels: ['Diabetes', 'Heart Disease'],
        datasets: [
            {
                label: 'Predictions Over Time',
                data: [0, 0], 
                backgroundColor: 'rgba(153, 102, 255, 0.6)',
                borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
            ],
            borderWidth: 1
        }]
    }
});

const individualsCtx = document.getElementById('individualsChart').getContext('2d');
const individualsChart = new Chart(individualsCtx, {
    type: 'bar',
    data: {
        labels: ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5'],
        datasets: [{
            label: 'Prediction Scores',
            data: [75, 85, 95, 65, 80], // Sample data
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});