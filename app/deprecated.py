
@app.route('/predict', methods=['POST'])
def api_predict():
    # First, getting the image into a usable format. Credit:
    # https://gist.github.com/mjul/32d697b734e7e9171cdb

    # Get the uploaded image
    img = request.files['file']

    # Store in memory
    in_memory_file = io.BytesIO()
    img.save(in_memory_file)

    # Convert to array representing black-and-white image
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)/255

    data = preprocess(img)

    # Reshape for the model
    #data = img.reshape(-1, 28*28)

    # The model is trained on white-on-black images. If a white-on-black image
    # is used, it will likely give incorrect results. Here is a basic fix to
    # try to detect black-on-white images and convert them. Probably not fool-proof.
    if (data.mean() > 0.5):
        data = value_invert(data)

    data = data.reshape((1, 1, 28, 28))
    # Generate prediction
    with graph.as_default():
        pred = model.predict_classes(data)[0]

    # Provide response
    return "The image you uploaded shows a " + str(pred) + ".\n"