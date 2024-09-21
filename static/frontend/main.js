let model;

const modelURL = 'http://localhost:5000/model';

const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const clearButton = document.getElementById("clear");
const numberOfFiles = document.getElementById("numberOfFiles");
const fileInput = document.getElementById('file');
const previewimg = document.getElementById("previewimg");

const loadModel = async (url) => {
    try {
        const loadedModel = await tf.loadLayersModel(url);
        return loadedModel;
    } catch (error) {
        console.error("Error loading model: ", error);
    }
};

const predict1 = async (modelURL) => {
    if (!model) model = await loadModel(modelURL);

    const files = fileInput.files;

    for (const img of files) {
        const data = new FormData();
        data.append('file', img);

        const processedImage = await fetch("/api/prepare", {
            method: 'POST',
            body: data
        })
        .then(response => response.json())
        .then(result => tf.tensor(result['images']));

        // Shape has to be the same as it was for training of the model
        const prediction = model.predict(tf.reshape(processedImage, [1, 48, 48, 3]));

        const label1 = prediction.argMax(axis = 1).dataSync()[0];
        const confidence = prediction.max(axis = 1).dataSync()[0];

        renderImageLabel(img, label1, confidence);
    }
};

const renderImageLabel = (img, label1, confidence) => {
    const reader = new FileReader();
    reader.onload = () => {
        const confidencePercentage = (confidence * 100).toFixed(2);
        const labelText = (label1 === 1) ? "Nam" : "Nữ";

        preview.innerHTML += `
            <div class="image-block">
                <img src="${reader.result}" class="image-block_loaded" id="source"/>
                <div class="label">
                    <div class="text">${labelText}</div>
                    <div class="per">${confidencePercentage}%</div>
                </div>
                
            </div>
            
            `
            ;
    };
    reader.readAsDataURL(img);
};

fileInput.addEventListener("change", () => {
    numberOfFiles.innerHTML = `Chọn ${fileInput.files.length} ảnh`;

    // Xóa preview cũ
    previewimg.innerHTML = "";

    // Lặp qua từng tệp đã được chọn
    for (const file of fileInput.files) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement("img");
            img.src = e.target.result;
            previewimg.appendChild(img);
        };
        reader.readAsDataURL(file);
    }
}, false);

predictButton.addEventListener("click", () => predict1(modelURL));
clearButton.addEventListener("click", () => {
    fileInput.value = "";
    preview.innerHTML = "";
    previewimg.innerHTML = "";
    numberOfFiles.innerHTML = "";
}, false);