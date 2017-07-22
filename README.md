Simple Flask API and canvas page to let the user interact with a neural network trained on the MNIST dataset. Its architecture is simply that from the [Keras example](https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py). A live version of the app is running over at [mnist-canvas.herokuapp.com](https://mnist-canvas.herokuapp.com/).

### File descriptions:

`server.py` (for Heroku) or `server_local.py` serves a Flask web app that loads a pre-trained neural network into Keras. It provides two ways of communicating with the backend:
 - Send it an image file on `/predict` e.g. `curl -X POST https://mnist-canvas.herokuapp.com/predict -F "file=@9.png"`
 - Send it a data url on `/post-data-url`, e.g. `curl -X POST https://mnist-canvas.herokuapp.com/post-data-url --data-urlencode "data=data:image/png;base64,iVBORw0KGgoAA....."`

On `/`, the server serves `/static/interface.html`, which provides a simple canvas interface where the user can draw a number and have the neural network try to recognize it. The Javascript on the page communicates with the backend via the data-url approach.

`generate_model.py` is the script that was used to train the neural network.

`MNIST_NN.h5` contains the saved neural network that the server loads up.

`Procfile`, `requirements.txt` and `runtime.txt` are architectural files for Heroku.

The image files are sample files that can be fed to the app.
