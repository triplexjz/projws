async function fetchData() {
    const response = await fetch('https://api.thingspeak.com/channels/YOUR_CHANNEL_ID/feeds.json?results=1');
    const data = await response.json();
    const feed = data.feeds[0];
    document.getElementById('temperature').innerText = `Temperature: ${feed.field1} °C`;
    document.getElementById('humidity').innerText = `Humidity: ${feed.field2} %`;
    document.getElementById('pressure').innerText = `Pressure: ${feed.field3} hPa`;
}
async function fetchLocation() {
    const latitude = '13.7229936';
    const longitude = '100.7728991';
    const googleAPIKey = 'YOUR_GOOGLE_MAPS_API_KEY'; // replace with your actual API key
    const url = `https://maps.googleapis.com/maps/api/geocode/json?latlng=${latitude},${longitude}&key=${googleAPIKey}`;

    try {
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === "OK" && data.results.length > 0) {
            // Get the formatted address from the first result
            const locationName = data.results[0].formatted_address;
            document.getElementById('locationName').textContent = `Location: ${locationName}`;
        } else {
            document.getElementById('locationName').textContent = "Location: Unavailable";
        }
    } catch (error) {
        console.error("Failed to fetch location data:", error);
    }
}

// Call the function when the page loads
fetchLocation();


setInterval(fetchData, 15000);
fetchData();