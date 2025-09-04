// HUD menu toggle (cosmetic)
const toggle = document.getElementById('toggle')
if (toggle) {
    toggle.addEventListener('click', () => {
        toggle.classList.toggle('active')
    })
}

function updateImage() {
    const imageElement = document.getElementById('latest-image');
    const fallbackUrl = imageElement.getAttribute('data-fallback-url');  // Get the fallback URL from the attribute

    fetch('/latest-image')
        .then(response => response.text())
        .then(data => {
            if (data && data !== 'No image available') {
                imageElement.src = 'data:image/jpeg;base64,' + data;
            } else {
                imageElement.src = fallbackUrl;  // Use the fallback URL when no image is available
            }
        })
        .catch(error => {
            console.log("Error fetching the latest image:", error);
            imageElement.src = fallbackUrl;  // Use the fallback URL in case of error
        });
}


// Function to fetch vehicle data using AJAX
function fetchVehicleData() {
    $.ajax({
        url: "/vehicle-data",
        type: "GET",
        success: function(data) {
            // Clear the existing content
                $(".textboxcontainer").empty();
            // Loop through the data and add new entries
        if (data.length > 0) {
                data.forEach(function(item) {
                    const numberPlate = item[0];
                    const timestamp = item[1];
                    $(".textboxcontainer").append(`
                        <div class="textbox">
                            <b>Number Plate:</b> ${numberPlate} <br>
                            <b>Timestamp:</b> ${timestamp} <br>
                        </div>
                    `);
                });
            } else {
                $(".textboxcontainer").append('<div class="textbox">No vehicle data available.</div>');
            }
        },
        error: function(error) {
            console.log("Error fetching vehicle data:", error);
        }
    });
}


// Update and Fetch vehicle data every 10 seconds
setInterval(() => {
    updateImage();
    fetchVehicleData();
}, 10000);


