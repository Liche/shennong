import json

from flask import Flask, jsonify, request

from tongue_classifier import classifier


app = Flask(__name__)

classifier.initialize()


@app.route('/classify', methods=['POST'])
def post():
    result = []

    urls = json.loads(request.data)

    predictions = classifier.classify(urls)

    for key, prediction in enumerate(predictions):
        result.append({
            'image': urls[key],
            'is_tongue': bool(prediction[0] == 0),
            'accuracy': prediction[1],

        })

    return jsonify(result)


if __name__ == '__main__':
    app.run()
