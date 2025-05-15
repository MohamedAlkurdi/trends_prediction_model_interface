from flask import Flask, request, jsonify
import os
import sys

# Add the project root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# Import your model logic
from models.build_country_trend_model import build_country_topic_model

app = Flask(__name__)

@app.route('/country_trend_model', methods=['POST'])
def get_country_trend_model():
    try:
        data = request.get_json()
        country = data.get('country')
        topic = data.get('topic')
        if not country or not topic:
            return jsonify({'error': 'Missing country or topic in request'}), 400
        result = build_country_topic_model(country, topic)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)