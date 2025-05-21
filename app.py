# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import logging
# from werkzeug.utils import secure_filename
# from datetime import datetime


# from model import recommend_furniture  # Your recommendation model

# # Initialize Flask appip
# app = Flask(__name__)
# CORS(app)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Configuration
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def health_check():
#     return jsonify({
#         'status': 'running',
#         'message': 'Furniture Recommendation API',
#         'timestamp': datetime.now().isoformat()
#     })

# @app.route('/recommend', methods=['GET', 'POST'])
# def recommend():
#     try:
#         logger.info("Received recommendation request")
        
#         # Validate request content type
#         if 'multipart/form-data' not in request.content_type:
#             logger.error(f"Invalid content type: {request.content_type}")
#             return jsonify({'error': 'Content-Type must be multipart/form-data'}), 400

#         # Get form data
#         category = request.form.get('category')
#         if not category:
#             logger.error("Missing 'category' in form data")
#             return jsonify({'error': 'Category is required'}), 400

#         # Validate image file
#         if 'image' not in request.files:
#             logger.error("No 'image' file part in request")
#             return jsonify({'error': 'No image provided'}), 400
            
#         image = request.files['image']
#         if image.filename == '':
#             logger.error("Empty filename")
#             return jsonify({'error': 'No selected file'}), 400
            
#         if not allowed_file(image.filename):
#             logger.error(f"Invalid file type: {image.filename}")
#             return jsonify({'error': 'Allowed file types are: png, jpg, jpeg, gif'}), 400

#         # Save uploaded file with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{timestamp}_{secure_filename(image.filename)}"
#         image_path = os.path.join(UPLOAD_FOLDER, filename)
#         image.save(image_path)
#         logger.info(f"Image temporarily saved to: {image_path}")

#         # Get recommendations
#         logger.info(f"Processing recommendation for category: {category}")
#         recommendations = recommend_furniture(image_path, category)
#         logger.info(f"Generated recommendations: {recommendations}")

#         # Clean up temporary file
#         try:
#             os.remove(image_path)
#             logger.info(f"Removed temporary file: {image_path}")
#         except Exception as e:
#             logger.warning(f"Could not remove temporary file: {str(e)}")

#         return jsonify({
#             'status': 'success',
#             'message': f'Recommendations for {category}',
#             'recommended': recommendations,
#             'processed_at': datetime.now().isoformat()
#         })

#     except Exception as e:
#         logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
#         return jsonify({
#             'status': 'error',
#             'message': 'Internal server error',
#             'error': str(e)
#         }), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)



from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
from room_classifier import is_room_image
from model import recommend_furniture  # Your custom model logic

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def health_check():
    return jsonify({
        'status': 'running',
        'message': 'Furniture Recommendation API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        logger.info("POST /recommend called")

        # Validate form inputs
        category = request.form.get('category')
        image_file = request.files.get('image')

        if not category:
            return jsonify({'error': 'Missing category'}), 400
        if not image_file or image_file.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(image_file.filename)}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        # Check if it's a valid room image
        if not is_room_image(image_path):
            os.remove(image_path)
            return jsonify({'error': 'Please upload a valid room photo (not object-only)'}), 400

        # Recommend furniture
        recommendations = recommend_furniture(image_path, category)

        # Cleanup
        os.remove(image_path)

        return jsonify({
            'status': 'success',
            'message': f'Recommendations for {category}',
            'recommended': recommendations,
            'processed_at': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error during recommendation: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
