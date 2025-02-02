document.getElementById("upload-form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("image-input");
    if (!fileInput.files.length) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to process the image.");
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error(error);
        alert("An error occurred while processing the image.");
    }
});

function displayResults(data) {
    document.getElementById("detected_plate_txt").textContent = data.detected_plate_txt;
    document.getElementById("detected_chars_image").src = data.detected_chars_image;
    document.getElementById("detected_plate_image").src = data.detected_plate_image;
}
