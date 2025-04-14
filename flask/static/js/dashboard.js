document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    const tabs = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Camera functionality elements
    const entryCameraToggle = document.getElementById('entryCameraToggle');
    const entryCameraContainer = document.getElementById('entryCameraContainer');
    const entryCameraView = document.getElementById('entryCameraView');
    const entryCaptureBtn = document.getElementById('entryCaptureBtn');
    const entryCaptureCanvas = document.getElementById('entryCaptureCanvas');
    const entryCapturePreview = document.getElementById('entryCapturePreview');
    const entryFileGroup = document.getElementById('entryFileGroup');
    const recognizeEntryBtn = document.getElementById('recognizeEntryBtn');
    const entryAutoDetectionToggle = document.getElementById('entryAutoDetectionToggle');
    
    const exitCameraToggle = document.getElementById('exitCameraToggle');
    const exitCameraContainer = document.getElementById('exitCameraContainer');
    const exitCameraView = document.getElementById('exitCameraView');
    const exitCaptureBtn = document.getElementById('exitCaptureBtn');
    const exitCaptureCanvas = document.getElementById('exitCaptureCanvas');
    const exitCapturePreview = document.getElementById('exitCapturePreview');
    const exitFileGroup = document.getElementById('exitFileGroup');
    const recognizeExitBtn = document.getElementById('recognizeExitBtn');
    const exitAutoDetectionToggle = document.getElementById('exitAutoDetectionToggle');
    
    let entryStream = null;
    let exitStream = null;
    
    // Auto detection variables
    let entryAutoDetectionActive = false;
    let exitAutoDetectionActive = false;
    let entryAutoDetectionTimer = null;
    let exitAutoDetectionTimer = null;
    let entryProcessingInProgress = false;
    let exitProcessingInProgress = false;
    let entryLastDetectionTime = 0;
    let exitLastDetectionTime = 0;
    const detectionCooldown = 5000; // 5 seconds cooldown between detections
    
    // Set up camera toggles if elements exist
    if (entryCameraToggle) {
        entryCameraToggle.addEventListener('change', function() {
            if (this.checked) {
                entryCameraContainer.style.display = 'block';
                entryFileGroup.style.display = 'none';
                startCamera(entryCameraView, stream => {
                    entryStream = stream;
                    
                    // No longer start auto detection automatically
                    // Auto detection will only start when the toggle is checked
                });
            } else {
                entryCameraContainer.style.display = 'none';
                entryFileGroup.style.display = 'flex';
                stopCamera(entryStream);
                entryCapturePreview.style.display = 'none';
                
                // Stop auto detection if active
                stopAutoDetection('entry');
            }
        });
    }
    
    // Set up auto detection toggles
    if (entryAutoDetectionToggle) {
        entryAutoDetectionToggle.addEventListener('change', function() {
            if (this.checked) {
                startAutoDetection('entry');
                document.getElementById('entryResults').innerHTML = 
                    '<div class="alert alert-info">Detección automática activada. Apunte la cámara hacia una matrícula.</div>';
            } else {
                stopAutoDetection('entry');
                document.getElementById('entryResults').innerHTML = '';
            }
        });
    }
    
    if (exitCameraToggle) {
        exitCameraToggle.addEventListener('change', function() {
            if (this.checked) {
                exitCameraContainer.style.display = 'block';
                exitFileGroup.style.display = 'none';
                startCamera(exitCameraView, stream => {
                    exitStream = stream;
                    
                    // No longer start auto detection automatically
                    // Auto detection will only start when the toggle is checked
                });
            } else {
                exitCameraContainer.style.display = 'none';
                exitFileGroup.style.display = 'flex';
                stopCamera(exitStream);
                exitCapturePreview.style.display = 'none';
                
                // Stop auto detection if active
                stopAutoDetection('exit');
            }
        });
    }
    
    // Set up auto detection toggles
    if (exitAutoDetectionToggle) {
        exitAutoDetectionToggle.addEventListener('change', function() {
            if (this.checked) {
                startAutoDetection('exit');
                document.getElementById('exitResults').innerHTML = 
                    '<div class="alert alert-info">Detección automática activada. Apunte la cámara hacia una matrícula.</div>';
            } else {
                stopAutoDetection('exit');
                document.getElementById('exitResults').innerHTML = '';
            }
        });
    }
    
    // Set up capture buttons (still keep them for manual capture if needed)
    if (entryCaptureBtn) {
        entryCaptureBtn.addEventListener('click', function() {
            captureImage(entryCameraView, entryCaptureCanvas, entryCapturePreview);
            
            // Auto-recognize after capture
            setTimeout(() => {
                const imgDataUrl = entryCapturePreview.src;
                processLicensePlate(imgDataUrl, 'entry', true);
            }, 500);
        });
    }
    
    if (exitCaptureBtn) {
        exitCaptureBtn.addEventListener('click', function() {
            captureImage(exitCameraView, exitCaptureCanvas, exitCapturePreview);
            
            // Auto-recognize after capture
            setTimeout(() => {
                const imgDataUrl = exitCapturePreview.src;
                processLicensePlate(imgDataUrl, 'exit', true);
            }, 500);
        });
    }
    
    // Auto detection functions
    function startAutoDetection(type) {
        if (type === 'entry') {
            if (entryAutoDetectionActive) return;
            
            entryAutoDetectionActive = true;
            entryAutoDetectionTimer = setInterval(() => {
                if (entryProcessingInProgress) return;
                
                // Check cooldown
                const now = Date.now();
                if (now - entryLastDetectionTime < detectionCooldown) return;
                
                // Capture and process frame
                captureImage(entryCameraView, entryCaptureCanvas, entryCapturePreview, false);
                const imgDataUrl = entryCaptureCanvas.toDataURL('image/png');
                processFrameForAutoDetection(imgDataUrl, 'entry');
            }, 1000); // Check every second
        } else if (type === 'exit') {
            if (exitAutoDetectionActive) return;
            
            exitAutoDetectionActive = true;
            exitAutoDetectionTimer = setInterval(() => {
                if (exitProcessingInProgress) return;
                
                // Check cooldown
                const now = Date.now();
                if (now - exitLastDetectionTime < detectionCooldown) return;
                
                // Capture and process frame
                captureImage(exitCameraView, exitCaptureCanvas, exitCapturePreview, false);
                const imgDataUrl = exitCaptureCanvas.toDataURL('image/png');
                processFrameForAutoDetection(imgDataUrl, 'exit');
            }, 1000); // Check every second
        }
    }
    
    function stopAutoDetection(type) {
        if (type === 'entry') {
            entryAutoDetectionActive = false;
            if (entryAutoDetectionTimer) {
                clearInterval(entryAutoDetectionTimer);
                entryAutoDetectionTimer = null;
            }
        } else if (type === 'exit') {
            exitAutoDetectionActive = false;
            if (exitAutoDetectionTimer) {
                clearInterval(exitAutoDetectionTimer);
                exitAutoDetectionTimer = null;
            }
        }
    }
    
    function processFrameForAutoDetection(imageData, type) {
        // Set flag to prevent multiple simultaneous processes
        if (type === 'entry') {
            entryProcessingInProgress = true;
        } else {
            exitProcessingInProgress = true;
        }
        
        // First detect if there's a license plate in the image
        const base64Data = imageData.split(',')[1];
        const blob = b64toBlob(base64Data, 'image/png');
        const formData = new FormData();
        formData.append('image', blob);
        
        fetch('/detectar_matricula', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // If a license plate was successfully detected (not just falling back to the full image)
            if (data.message && data.message.includes("Imagen recortada guardada")) {
                // Update last detection time
                if (type === 'entry') {
                    entryLastDetectionTime = Date.now();
                    
                    // Display the captured image
                    entryCapturePreview.src = 'data:image/jpeg;base64,' + data.image_data;
                    entryCapturePreview.style.display = 'block';
                } else {
                    exitLastDetectionTime = Date.now();
                    
                    // Display the captured image
                    exitCapturePreview.src = 'data:image/jpeg;base64,' + data.image_data;
                    exitCapturePreview.style.display = 'block';
                }
                
                // Process the license plate
                processLicensePlate('data:image/jpeg;base64,' + data.image_data, type, true);
                
                // Play a sound to indicate detection
                playDetectionSound();
            }
        })
        .catch(error => {
            console.error('Auto detection error:', error);
        })
        .finally(() => {
            // Reset processing flag
            if (type === 'entry') {
                entryProcessingInProgress = false;
            } else {
                exitProcessingInProgress = false;
            }
        });
    }
    
    function playDetectionSound() {
        // Create and play a beep sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.type = 'sine';
        oscillator.frequency.value = 830;
        gainNode.gain.value = 0.1;
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.15);
    }
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to current tab
            this.classList.add('active');
            
            // Hide all tab contents
            tabContents.forEach(content => content.classList.add('d-none'));
            
            // Show selected tab content
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId + 'Tab').classList.remove('d-none');
            
            // Special actions for specific tabs
            if (tabId === 'vehicles') {
                loadVehicles();
            } else if (tabId === 'settings') {
                loadConfig();
            } else if (tabId === 'main') {
                loadStatistics();
            }
        });
    });
    
    // Set up event listeners
    document.getElementById('recognizeEntryBtn')?.addEventListener('click', function() {
        processLicensePlate(null, 'entry', false);
    });
    
    document.getElementById('recognizeExitBtn')?.addEventListener('click', function() {
        processLicensePlate(null, 'exit', false);
    });
    
    document.getElementById('processPaymentBtn')?.addEventListener('click', processPayment);
    
    document.getElementById('configForm')?.addEventListener('submit', function(e) {
        e.preventDefault();
        saveConfig();
    });
    
    document.getElementById('applyFiltersBtn')?.addEventListener('click', function() {
        loadVehicles(1);
    });
    
    // Load initial statistics
    loadStatistics();
    
    // Refresh statistics every 30 seconds
    setInterval(loadStatistics, 30000);
});

// Camera functionality
// Start camera function
function startCamera(videoElement, callback) {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }, // prefer rear camera if available
            audio: false
        })
        .then(function(stream) {
            videoElement.srcObject = stream;
            if (callback) callback(stream);
        })
        .catch(function(error) {
            console.error("Camera error:", error);
            alert("No se pudo acceder a la cámara. Por favor, asegúrese de que tiene una cámara conectada y ha dado permisos.");
        });
    } else {
        alert("Su navegador no soporta el acceso a la cámara");
    }
}

// Stop camera function
function stopCamera(stream) {
    if (stream) {
        stream.getTracks().forEach(track => {
            track.stop();
        });
    }
}

// Capture image function
function captureImage(videoElement, canvasElement, previewElement) {
    // Set canvas dimensions to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    // Draw video frame to canvas
    const ctx = canvasElement.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // Get image data URL and display in preview
    const imageDataURL = canvasElement.toDataURL('image/png');
    previewElement.src = imageDataURL;
    previewElement.style.display = 'block';
}

// Unified function to process license plate from any source (camera or file)
function processLicensePlate(imageData, type, fromCamera = false) {
    const resultsDiv = type === 'entry' ? document.getElementById('entryResults') : document.getElementById('exitResults');
    resultsDiv.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p>Procesando imagen...</p></div>';
    
    let formData = new FormData();
    let fetchOptions = {};
    
    if (fromCamera) {
        // Handle image from camera (base64)
        const base64Data = imageData.split(',')[1];
        const blob = b64toBlob(base64Data, 'image/png');
        formData.append('image', blob);
        fetchOptions = {
            method: 'POST',
            body: formData
        };
    } else {
        // Handle image from file input
        const fileInput = document.getElementById(type === 'entry' ? 'entryImageInput' : 'exitImageInput');
        const file = fileInput.files[0];
        
        if (!file) {
            showAlert(resultsDiv.id, 'Por favor, selecciona una imagen', 'danger');
            return;
        }
        
        formData.append('image', file);
        fetchOptions = {
            method: 'POST',
            body: formData
        };
    }
    
    // First detect the license plate
    fetch('/detectar_matricula', fetchOptions)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Check if a license plate was detected or not
        const wasPlateDetected = data.message && data.message.includes("Imagen recortada guardada");
        
        if (!wasPlateDetected) {
            // Add warning that plate wasn't detected automatically
            console.warn("No se detectó matrícula automáticamente, intentando con la imagen completa");
        }
        
        // Use the image data returned by detectar_matricula
        if (data.image_data) {
            // Now predict the characters and register using the detected plate image
            return fetch(`/predict?register=${type}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_data: data.image_data })
            });
        } else {
            // Fallback to the direct method if image_data is not available
            return fetch(`/predict?register=${type}`, fetchOptions);
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showAlert(resultsDiv.id, data.error, 'danger');
            return;
        }
        
        let html = '';
        
        if (data.license_plate) {
            // Display license plate
            html += `
                <div class="text-center mb-3">
                    <h4>Matrícula detectada:</h4>
                    <div class="license-plate">${data.license_plate}</div>
                </div>
            `;
            
            // Show characters with confidence
            if (data.plate_recognition && data.plate_recognition.length > 0) {
                html += '<div class="row justify-content-center mb-3">';
                data.plate_recognition.forEach(char => {
                    html += `
                        <div class="col-auto text-center mx-1">
                            <strong class="recognition-result">${char.character}</strong><br>
                            <small class="confidence">${char.confidence}</small>
                        </div>
                    `;
                });
                html += '</div>';
            }
            
            // Entry or exit success data
            if (type === 'entry' && data.entry_data) {
                if (data.entry_data.error) {
                    html += `<div class="alert alert-danger">${data.entry_data.error}</div>`;
                } else {
                    html += `
                        <div class="alert alert-success">
                            <h5 class="alert-heading">¡Entrada registrada correctamente!</h5>
                            <div class="ticket">
                                <p class="ticket-id">Ticket: ${data.entry_data.ticket_id}</p>
                                <p><strong>Matrícula:</strong> ${data.entry_data.license_plate}</p>
                                <p><strong>Hora de entrada:</strong> ${data.entry_data.entry_time}</p>
                            </div>
                        </div>
                    `;
                }
            } else if (type === 'exit' && data.exit_data) {
                if (data.exit_data.error) {
                    html += `<div class="alert alert-danger">${data.exit_data.error}</div>`;
                } else {
                    html += `
                        <div class="alert alert-info">
                            <h5 class="alert-heading">¡Salida registrada correctamente!</h5>
                            <div class="ticket">
                                <p class="ticket-id">Ticket: ${data.exit_data.ticket_id}</p>
                                <p><strong>Matrícula:</strong> ${data.exit_data.license_plate}</p>
                                <p><strong>Hora de entrada:</strong> ${data.exit_data.entry_time}</p>
                                <p><strong>Hora de salida:</strong> ${data.exit_data.exit_time}</p>
                                <p><strong>Duración:</strong> ${data.exit_data.duration_hours.toFixed(2)} horas</p>
                                <p><strong>Importe:</strong> ${data.exit_data.amount.toFixed(2)} €</p>
                            </div>
                        </div>
                    `;
                }
            }
        } else {
            html = `<div class="alert alert-warning">No se pudo detectar ninguna matrícula. Por favor, intenta con otra imagen o asegúrate que la matrícula sea visible.</div>`;
        }
        
        resultsDiv.innerHTML = html;
        
        // Reload statistics
        loadStatistics();
    })
    .catch(error => {
        console.error('Error:', error);
        resultsDiv.innerHTML = `<div class="alert alert-danger">Error al procesar la imagen: ${error.message}</div>`;
    });
}

// Helper function to convert base64 to blob
function b64toBlob(b64Data, contentType = '', sliceSize = 512) {
    const byteCharacters = atob(b64Data);
    const byteArrays = [];
    
    for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
        const slice = byteCharacters.slice(offset, offset + sliceSize);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
    }
    
    return new Blob(byteArrays, { type: contentType });
}

// Function to process payment
function processPayment() {
    const ticketId = document.getElementById('ticketIdInput').value.trim();
    
    if (!ticketId) {
        showAlert('paymentResults', 'Por favor, introduce el ID del ticket', 'danger');
        return;
    }
    
    // Show loading
    document.getElementById('paymentResults').innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Procesando...</span>
            </div>
            <p class="mt-2">Procesando pago...</p>
        </div>
    `;
    
    fetch('/payment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ ticket_id: ticketId })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            showAlert('paymentResults', data.error, 'danger');
            return;
        }
        
        let html = `
            <div class="alert alert-success">
                <h5 class="alert-heading">¡Pago procesado correctamente!</h5>
                <p><strong>Matrícula:</strong> ${data.license_plate}</p>
                <p><strong>Importe:</strong> ${data.amount_paid ? data.amount_paid.toFixed(2) + ' €' : 'N/A'}</p>
                <p><strong>Ticket:</strong> ${data.ticket_id}</p>
                <hr>
                <div class="time-warning">
                    <i class="fas fa-clock me-1"></i>
                    <span>Dispone de <strong>${data.exit_time_window} minutos</strong> para salir del parking.</span>
                    <p class="small mb-0">Tiempo límite: ${data.exit_limit_time}</p>
                </div>
            </div>
        `;
        
        document.getElementById('paymentResults').innerHTML = html;
        
        // Reset input
        document.getElementById('ticketIdInput').value = '';
        
        // Reload statistics
        loadStatistics();
    })
    .catch(error => {
        showAlert('paymentResults', `Error: ${error.message}`, 'danger');
    });
}

// Function to load statistics
function loadStatistics() {
    fetch('/admin/statistics')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const statsDiv = document.getElementById('statisticsPanel');
        
        statsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <div class="stats-box bg-primary bg-opacity-10">
                        <p class="stats-value">${data.current_vehicles}</p>
                        <p class="stats-label">Vehículos actuales</p>
                    </div>
                </div>
                <div class="col-md-6 mb-3">
                    <div class="stats-box bg-success bg-opacity-10">
                        <p class="stats-value">${data.total_revenue.toFixed(2)} €</p>
                        <p class="stats-label">Ingresos totales</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <div class="stats-box bg-info bg-opacity-10">
                        <p class="stats-value">${data.entries_today}</p>
                        <p class="stats-label">Entradas hoy</p>
                    </div>
                </div>
                <div class="col-md-6 mb-3">
                    <div class="stats-box bg-danger bg-opacity-10">
                        <p class="stats-value">${data.exits_today}</p>
                        <p class="stats-label">Salidas hoy</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-12">
                    <div class="stats-box bg-secondary bg-opacity-10">
                        <p class="stats-value">${data.completed_visits}</p>
                        <p class="stats-label">Visitas completadas</p>
                    </div>
                </div>
            </div>
        `;
    })
    .catch(error => {
        console.error('Error loading statistics:', error);
        document.getElementById('statisticsPanel').innerHTML = `
            <div class="alert alert-danger">
                Error al cargar las estadísticas: ${error.message}
            </div>
        `;
    });
}

// Function to load vehicles with pagination
function loadVehicles(page = 1) {
    // Get filters
    const statusFilter = document.getElementById('vehicleStatusFilter').value;
    const paidFilter = document.getElementById('vehiclePaidFilter').value;
    
    // Build query parameters
    let params = new URLSearchParams({
        page: page,
        per_page: 10
    });
    
    if (statusFilter !== 'all') {
        params.append('status', statusFilter);
    }
    
    if (paidFilter !== 'all') {
        params.append('paid', paidFilter);
    }
    
    // Show loading
    document.getElementById('vehiclesTableBody').innerHTML = `
        <tr>
            <td colspan="8" class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Cargando...</span>
                </div>
                <p class="mt-2">Cargando vehículos...</p>
            </td>
        </tr>
    `;
    
    fetch('/vehicles?' + params.toString())
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const tableBody = document.getElementById('vehiclesTableBody');
        const pagination = document.getElementById('vehiclesPagination');
        
        if (data.vehicles.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center py-4">
                        No hay vehículos que coincidan con los filtros.
                    </td>
                </tr>
            `;
            pagination.innerHTML = '';
            return;
        }
        
        let html = '';
        
        data.vehicles.forEach(vehicle => {
            // Calculate duration if exit_time exists
            let durationText = 'En parking';
            if (vehicle.exit_time) {
                // Calculate hours between entry_time and exit_time
                const entryTime = new Date(vehicle.entry_time);
                const exitTime = new Date(vehicle.exit_time);
                const durationHours = (exitTime - entryTime) / (1000 * 60 * 60);
                durationText = durationHours.toFixed(2) + ' h';
            }
            
            // Status badge
            let statusBadge = '';
            if (vehicle.exit_time === null) {
                statusBadge = '<span class="badge bg-info">En parking</span>';
            } else if (vehicle.paid) {
                statusBadge = '<span class="badge bg-success">Pagado</span>';
            } else {
                statusBadge = '<span class="badge bg-warning text-dark">Pendiente</span>';
            }
            
            html += `
                <tr>
                    <td>${vehicle.license_plate}</td>
                    <td>${vehicle.ticket_id}</td>
                    <td>${vehicle.entry_time}</td>
                    <td>${vehicle.exit_time || '-'}</td>
                    <td>${durationText}</td>
                    <td>${vehicle.amount ? vehicle.amount.toFixed(2) + ' €' : '-'}</td>
                    <td>${statusBadge}</td>
                    <td>
                        <button class="btn btn-sm btn-info btn-action" 
                                onclick="verifyTicket('${vehicle.ticket_id}')">
                            <i class="fas fa-info-circle"></i>
                        </button>
                        ${!vehicle.paid && vehicle.exit_time ? 
                            `<button class="btn btn-sm btn-success btn-action ms-1" 
                                onclick="payVehicle('${vehicle.ticket_id}')">
                                <i class="fas fa-money-bill-wave"></i>
                            </button>` 
                            : ''}
                    </td>
                </tr>
            `;
        });
        
        tableBody.innerHTML = html;
        
        // Build pagination
        let paginationHtml = '';
        
        // Previous button
        paginationHtml += `
            <li class="page-item ${data.page === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="loadVehicles(${data.page - 1}); return false;">
                    <i class="fas fa-chevron-left"></i>
                </a>
            </li>
        `;
        
        // Page numbers
        for (let i = 1; i <= data.pages; i++) {
            paginationHtml += `
                <li class="page-item ${i === data.page ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="loadVehicles(${i}); return false;">
                        ${i}
                    </a>
                </li>
            `;
        }
        
        // Next button
        paginationHtml += `
            <li class="page-item ${data.page === data.pages || data.pages === 0 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="loadVehicles(${data.page + 1}); return false;">
                    <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;
        
        pagination.innerHTML = paginationHtml;
    })
    .catch(error => {
        document.getElementById('vehiclesTableBody').innerHTML = `
            <tr>
                <td colspan="8" class="text-center py-4">
                    <div class="alert alert-danger mb-0">
                        Error al cargar los vehículos: ${error.message}
                    </div>
                </td>
            </tr>
        `;
        document.getElementById('vehiclesPagination').innerHTML = '';
    });
}

// Function to verify a ticket
function verifyTicket(ticketId) {
    fetch(`/verify_ticket/${ticketId}`)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }
        
        // Create a modal to display ticket details
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'ticketModal';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-hidden', 'true');
        
        let statusClass, statusText;
        if (data.exit_time === null) {
            statusClass = 'bg-info';
            statusText = 'En parking';
        } else if (data.paid) {
            statusClass = 'bg-success';
            statusText = 'Pagado';
        } else {
            statusClass = 'bg-warning text-dark';
            statusText = 'Pendiente de pago';
        }
        
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Detalles del Ticket</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="ticket mb-3">
                            <p class="ticket-id">Ticket: ${data.ticket_id}</p>
                            <div class="row mb-2">
                                <div class="col-6"><strong>Matrícula:</strong></div>
                                <div class="col-6">${data.license_plate}</div>
                            </div>
                            <div class="row mb-2">
                                <div class="col-6"><strong>Entrada:</strong></div>
                                <div class="col-6">${data.entry_time}</div>
                            </div>
                            <div class="row mb-2">
                                <div class="col-6"><strong>Salida:</strong></div>
                                <div class="col-6">${data.exit_time || 'En parking'}</div>
                            </div>
                            ${data.amount ? `
                            <div class="row mb-2">
                                <div class="col-6"><strong>Importe:</strong></div>
                                <div class="col-6">${data.amount.toFixed(2)} €</div>
                            </div>` : ''}
                            <div class="row">
                                <div class="col-6"><strong>Estado:</strong></div>
                                <div class="col-6"><span class="badge ${statusClass}">${statusText}</span></div>
                            </div>
                        </div>
                        ${data.exit_time && !data.paid ? `
                        <div class="text-center">
                            <button class="btn btn-success" onclick="payVehicle('${data.ticket_id}'); return false;">
                                <i class="fas fa-money-bill-wave me-1"></i> Procesar Pago
                            </button>
                        </div>` : ''}
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Show the modal
        const modalInstance = new bootstrap.Modal(document.getElementById('ticketModal'));
        modalInstance.show();
        
        // Remove modal from DOM after it's hidden
        document.getElementById('ticketModal').addEventListener('hidden.bs.modal', function() {
            document.body.removeChild(modal);
        });
    })
    .catch(error => {
        alert(`Error al verificar el ticket: ${error.message}`);
    });
}

// Function to pay for a vehicle
function payVehicle(ticketId) {
    // If modal is open, close it
    const modal = document.getElementById('ticketModal');
    if (modal) {
        const modalInstance = bootstrap.Modal.getInstance(modal);
        modalInstance.hide();
    }
    
    // Fill the ticket ID in the payment form and switch to the main tab
    document.getElementById('ticketIdInput').value = ticketId;
    
    // Switch to main tab
    document.querySelector('.nav-link[data-tab="main"]').click();
    
    // Scroll to the payment section
    document.getElementById('processPaymentBtn').scrollIntoView({
        behavior: 'smooth'
    });
    
    // Flash the payment form
    const paymentCard = document.getElementById('processPaymentBtn').closest('.card');
    paymentCard.classList.add('border-success');
    setTimeout(() => {
        paymentCard.classList.remove('border-success');
    }, 2000);
}

// Function to load configuration
function loadConfig() {
    fetch('/admin/config')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('ratePerHourInput').value = data.rate_per_hour;
        document.getElementById('maxCapacityInput').value = data.max_capacity;
    })
    .catch(error => {
        showAlert('configResults', `Error al cargar la configuración: ${error.message}`, 'danger');
    });
}

// Function to save configuration
function saveConfig() {
    const ratePerHour = document.getElementById('ratePerHourInput').value;
    const maxCapacity = document.getElementById('maxCapacityInput').value;
    
    if (!ratePerHour || !maxCapacity) {
        showAlert('configResults', 'Por favor, completa todos los campos', 'danger');
        return;
    }
    
    fetch('/admin/config', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            rate_per_hour: ratePerHour,
            max_capacity: maxCapacity
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        showAlert('configResults', 'Configuración guardada correctamente', 'success');
    })
    .catch(error => {
        showAlert('configResults', `Error al guardar la configuración: ${error.message}`, 'danger');
    });
}

// Helper function to show alerts
function showAlert(containerId, message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    container.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = bootstrap.Alert.getInstance(alertDiv);
        if (alert) {
            alert.close();
        }
    }, 5000);
}