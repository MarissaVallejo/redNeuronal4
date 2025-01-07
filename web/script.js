document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navButtons = document.querySelectorAll('.nav-button');
    const sections = document.querySelectorAll('.content-section');

    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const sectionId = button.dataset.section;
            
            // Update active states
            navButtons.forEach(btn => btn.classList.remove('active'));
            sections.forEach(section => section.classList.remove('active'));
            
            button.classList.add('active');
            document.getElementById(sectionId).classList.add('active');
        });
    });

    // Image Upload Preview
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');

    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    });

    // Video Upload Preview
    const videoUpload = document.getElementById('video-upload');
    const videoPreview = document.getElementById('video-preview');
    const videoContainer = document.getElementById('video-container');
    const frameSlider = document.getElementById('frame-slider');
    const frameInfo = document.getElementById('frame-info');

    videoUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            videoPreview.src = url;
            videoContainer.classList.remove('hidden');
            
            videoPreview.onloadedmetadata = function() {
                frameSlider.max = videoPreview.duration;
                frameInfo.textContent = `Duration: ${videoPreview.duration.toFixed(2)}s`;
            };
        }
    });

    frameSlider.addEventListener('input', function() {
        videoPreview.currentTime = frameSlider.value;
        frameInfo.textContent = `Time: ${frameSlider.value}s / ${videoPreview.duration.toFixed(2)}s`;
    });

    // Training Progress Simulation
    const trainButton = document.getElementById('train-button');
    const trainingProgress = document.getElementById('training-progress');
    const progressBar = document.querySelector('.progress');
    const progressText = document.querySelector('.progress-text');
    const trainingResults = document.getElementById('training-results');

    trainButton.addEventListener('click', function() {
        trainingProgress.classList.remove('hidden');
        let progress = 0;
        
        const interval = setInterval(() => {
            progress += 1;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `Progreso: ${progress}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
                trainingResults.classList.remove('hidden');
                createTrainingChart();
            }
        }, 100);
    });

    // Charts
    function createTrainingChart() {
        const ctx = document.getElementById('training-chart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: 20}, (_, i) => i + 1),
                datasets: [{
                    label: 'Precisión de Entrenamiento',
                    data: Array.from({length: 20}, () => Math.random() * 0.4 + 0.6),
                    borderColor: 'rgb(255, 75, 75)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    // Add new code for block analysis
    function initializeBlockAnalysis(imageData, blocksData) {
        const container = document.querySelector('.block-analysis-container');
        if (!container) return;

        const imageElement = container.querySelector('img');
        const blockColors = {
            'DIQUE': '#FF6B6B',
            'SPB': '#4ECDC4',
            'SPP': '#45B7D1',
            'SSM': '#96CEB4',
            'VM': '#FFBE0B',
            'VOLCANICO': '#FF006E'
        };

        // Create zoom container
        const zoomContainer = document.createElement('div');
        zoomContainer.className = 'zoom-container';
        document.body.appendChild(zoomContainer);

        function createBlockOverlays() {
            const imageWidth = imageElement.offsetWidth;
            const imageHeight = imageElement.offsetHeight;
            const rows = blocksData.grid_size[0];
            const cols = blocksData.grid_size[1];

            const blockWidth = imageWidth / cols;
            const blockHeight = imageHeight / rows;

            // Clear existing overlays
            const existingOverlays = container.querySelectorAll('.block-overlay');
            existingOverlays.forEach(overlay => overlay.remove());

            // Create new overlays
            blocksData.blocks.forEach((block, index) => {
                const row = Math.floor(index / cols);
                const col = index % cols;

                const overlay = document.createElement('div');
                overlay.className = 'block-overlay';
                overlay.style.width = `${blockWidth}px`;
                overlay.style.height = `${blockHeight}px`;
                overlay.style.left = `${col * blockWidth}px`;
                overlay.style.top = `${row * blockHeight}px`;
                overlay.style.backgroundColor = `${blockColors[block.class]}40`; // 40 for 25% opacity
                overlay.style.borderColor = blockColors[block.class];

                // Add hover handlers
                overlay.addEventListener('mouseenter', (e) => {
                    showBlockDetails(e, block);
                    showZoomView(e, imageData, row, col, rows, cols);
                });

                overlay.addEventListener('mouseleave', () => {
                    hideBlockDetails();
                    hideZoomView();
                });

                container.appendChild(overlay);
            });

            unifyAdjacentBlocks();
        }

        function showBlockDetails(event, blockData) {
            const details = document.createElement('div');
            details.className = 'block-details active';
            details.style.left = `${event.pageX + 10}px`;
            details.style.top = `${event.pageY + 10}px`;

            details.innerHTML = `
                <h3>Detalles del Bloque</h3>
                <p>Clase: ${blockData.class}</p>
                <p>Confianza: ${(blockData.confidence * 100).toFixed(2)}%</p>
            `;

            document.body.appendChild(details);
        }

        function hideBlockDetails() {
            const details = document.querySelector('.block-details');
            if (details) {
                details.remove();
            }
        }

        function showZoomView(event, imageData, row, col, rows, cols) {
            const zoomContainer = document.querySelector('.zoom-container');
            const zoomImage = document.createElement('img');
            zoomImage.className = 'zoom-image';

            // Calculate the portion of the image to zoom
            const blockWidth = imageData.width / cols;
            const blockHeight = imageData.height / rows;
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = blockWidth;
            canvas.height = blockHeight;
            ctx.drawImage(
                imageData,
                col * blockWidth, row * blockHeight,
                blockWidth, blockHeight,
                0, 0,
                blockWidth, blockHeight
            );

            zoomImage.src = canvas.toDataURL();

            zoomContainer.innerHTML = '';
            zoomContainer.appendChild(zoomImage);
            zoomContainer.style.display = 'block';
            zoomContainer.style.left = `${event.pageX + 20}px`;
            zoomContainer.style.top = `${event.pageY + 20}px`;
        }

        function hideZoomView() {
            const zoomContainer = document.querySelector('.zoom-container');
            if (zoomContainer) {
                zoomContainer.style.display = 'none';
            }
        }

        function unifyAdjacentBlocks() {
            const overlays = Array.from(container.querySelectorAll('.block-overlay'));
            const rows = blocksData.grid_size[0];
            const cols = blocksData.grid_size[1];

            overlays.forEach((overlay, index) => {
                const row = Math.floor(index / cols);
                const col = index % cols;
                const currentClass = blocksData.blocks[index].class;

                // Check right neighbor
                if (col < cols - 1) {
                    const rightNeighbor = overlays[index + 1];
                    const rightClass = blocksData.blocks[index + 1].class;
                    if (currentClass === rightClass) {
                        overlay.style.borderRight = 'none';
                        rightNeighbor.style.borderLeft = 'none';
                    }
                }

                // Check bottom neighbor
                if (row < rows - 1) {
                    const bottomNeighbor = overlays[index + cols];
                    const bottomClass = blocksData.blocks[index + cols].class;
                    if (currentClass === bottomClass) {
                        overlay.style.borderBottom = 'none';
                        bottomNeighbor.style.borderTop = 'none';
                    }
                }
            });
        }

        // Initialize block analysis when image is loaded
        if (imageElement.complete) {
            createBlockOverlays();
        } else {
            imageElement.onload = createBlockOverlays;
        }
    }

});