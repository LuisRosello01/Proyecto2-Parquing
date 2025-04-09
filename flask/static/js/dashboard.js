document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    const tabs = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');
    
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
    document.getElementById('recognizeEntryBtn').addEventListener('click', function() {
        recognizePlate('entryImageInput', 'entry');
    });
    
    document.getElementById('recognizeExitBtn').addEventListener('click', function() {
        recognizePlate('exitImageInput', 'exit');
    });
    
    document.getElementById('processPaymentBtn').addEventListener('click', processPayment);
    
    document.getElementById('configForm').addEventListener('submit', function(e) {
        e.preventDefault();
        saveConfig();
    });
    
    document.getElementById('applyFiltersBtn').addEventListener('click', function() {
        loadVehicles(1);
    });
    
    // Load initial statistics
    loadStatistics();
    
    // Refresh statistics every 30 seconds
    setInterval(loadStatistics, 30000);
});

// Function to recognize license plate
function recognizePlate(inputId, mode) {
    const fileInput = document.getElementById(inputId);
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert(mode === 'entry' ? 'entryResults' : 'exitResults', 'Por favor, selecciona una imagen', 'danger');
        return;
    }
    
    // Show loading
    const resultsDiv = document.getElementById(mode === 'entry' ? 'entryResults' : 'exitResults');
    resultsDiv.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Procesando...</span>
            </div>
            <p class="mt-2">Procesando imagen...</p>
        </div>
    `;
    
    const formData = new FormData();
    formData.append('image', file);
    
    // First detect the license plate
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
        
        // Usar la imagen base64 devuelta por detectar_matricula
        if (data.image_data) {
            // Now predict the characters and register using the detected plate image
            return fetch(`/predict?register=${mode}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_data: data.image_data })
            });
        } else {
            // Fallback to the old method if image_data is not available
            return fetch(`/predict?register=${mode}`, {
                method: 'POST',
                body: formData
            });
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
            showAlert(mode === 'entry' ? 'entryResults' : 'exitResults', data.error, 'danger');
            return;
        }
        
        let html = '';
        
        if (data.license_plate) {
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
        }
        
        if (mode === 'entry') {
            if (data.entry_data) {
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
            }
        } else if (mode === 'exit') {
            if (data.exit_data) {
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
                                <p><strong>Estado:</strong> <span class="badge bg-warning text-dark">Pendiente de pago</span></p>
                            </div>
                        </div>
                    `;
                }
            }
        }
        
        resultsDiv.innerHTML = html;
        
        // Reload statistics
        loadStatistics();
    })
    .catch(error => {
        showAlert(mode === 'entry' ? 'entryResults' : 'exitResults', `Error: ${error.message}`, 'danger');
    });
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