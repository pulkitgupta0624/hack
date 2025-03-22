from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import logging
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import os
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Image processing model with lazy initialization
image_model = None

def get_image_model():
    global image_model
    if image_model is None:
        try:
            logger.info("Initializing ResNet50 model")
            image_model = ResNet50(weights='imagenet')
            logger.info("ResNet50 model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing image model: {str(e)}")
            return None
    return image_model

def preprocess_image(image_data):
    try:
        # Convert base64 to image
        img = Image.open(io.BytesIO(image_data))
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_items(image_data):
    model = get_image_model()
    if model is None:
        return []
    
    img = preprocess_image(image_data)
    if img is None:
        return []
    
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=5)[0]
    return [(label, float(prob)) for (_, label, prob) in decoded_preds]

class SemanticNICCodeProcessor:
    def __init__(self, csv_path):
        logger.info("Initializing Semantic NIC Code Processor")
        
        # Load CSV with efficient dtype specifications
        start_time = time.time()
        try:
            self.df = pd.read_csv(csv_path, dtype={
                'Sub Class': 'int32',
                'Description': 'string', 
                'Division': 'string', 
                'Section': 'string'
            })
            logger.info(f"CSV loaded in {time.time() - start_time:.2f} seconds. Total rows: {len(self.df)}")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
        
        # Load pre-trained semantic search model
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading semantic model: {str(e)}")
            raise
        
        # Precompute embeddings for all descriptions
        self.precompute_embeddings()
        
        # Query cache to improve performance
        self.query_cache = {}
        self.max_cache_size = 100
    
    def precompute_embeddings(self):
        """Precompute embeddings for all NIC code descriptions"""
        logger.info("Precomputing embeddings for NIC code descriptions")
        start_time = time.time()
        
        # Combine description fields for richer embedding
        combined_descriptions = (
            self.df['Description'] + ' ' + 
            self.df['Division'] + ' ' + 
            self.df['Section']
        )
        
        # Compute embeddings
        self.description_embeddings = self.semantic_model.encode(
            combined_descriptions.tolist(), 
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        logger.info(f"Embeddings computed in {time.time() - start_time:.2f} seconds")
    
    def find_nic_codes(self, query, top_n=5):
        """Find NIC codes using semantic search"""
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.query_cache:
            logger.info("Using cached result")
            return self.query_cache[cache_key]
        
        # Encode query
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            self.description_embeddings
        )[0]
        
        # Find top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        results = []
        min_threshold = 0.1  # Adjust based on your data
        for idx in top_indices:
            if similarities[idx] > min_threshold:
                results.append({
                    'nic_code': int(self.df.iloc[idx]['Sub Class']),
                    'description': self.df.iloc[idx]['Description'],
                    'division': self.df.iloc[idx]['Division'],
                    'section': self.df.iloc[idx]['Section'],
                    'similarity_score': float(similarities[idx])
                })
        
        # Prepare response
        response = {
            'results': results,
            'query': query,
            'total_matches': len(results)
        }
        
        # Cache management
        if len(self.query_cache) >= self.max_cache_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = response
        
        logger.info(f"Found {len(results)} results in {time.time() - start_time:.2f} seconds")
        return response

# Global processor with lazy initialization
nic_processor = None
govt_schemes_df = None

def get_processor():
    global nic_processor
    if nic_processor is None:
        try:
            logger.info("Initializing Semantic NIC processor")
            nic_processor = SemanticNICCodeProcessor('nic_2008.csv')
            logger.info("Semantic NIC processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing processor: {str(e)}")
            return None
    return nic_processor

def get_govt_schemes():
    global govt_schemes_df
    if govt_schemes_df is None:
        try:
            logger.info("Loading government schemes CSV")
            govt_schemes_df = pd.read_csv('Cleaned_CenterSectorScheme2021-22 (1).csv')
            logger.info("Government schemes CSV loaded successfully")
        except Exception as e:
            logger.error(f"Error loading government schemes CSV: {str(e)}")
            return None
    return govt_schemes_df

def get_relevant_schemes(nic_code, description=None):
    """Fetch relevant government schemes based on NIC code and description using semantic similarity"""
    schemes_df = get_govt_schemes()
    processor = get_processor()  # Get access to the semantic model
    
    if schemes_df is None or processor is None:
        logger.error("Government schemes dataframe or processor is None")
        return []
    
    # If no description provided, try to find it based on NIC code
    if description is None:
        # Find the description for this NIC code from the NIC dataframe
        try:
            nic_df = processor.df
            nic_row = nic_df[nic_df['Sub Class'] == int(nic_code)]
            if not nic_row.empty:
                description = nic_row.iloc[0]['Description'] + ' ' + nic_row.iloc[0]['Division'] + ' ' + nic_row.iloc[0]['Section']
                logger.info(f"Found description for NIC code {nic_code}: {description}")
            else:
                logger.warning(f"No description found for NIC code {nic_code}")
                # Fall back to the original method
                return get_relevant_schemes_by_code(nic_code)
        except Exception as e:
            logger.error(f"Error finding description for NIC code: {str(e)}")
            return get_relevant_schemes_by_code(nic_code)
    
    # Use semantic search to find relevant schemes
    logger.info(f"Finding schemes relevant to description: {description}")
    
    # Get embeddings for the description
    query_embedding = processor.semantic_model.encode(description, convert_to_tensor=True)
    
    # Get embeddings for all scheme names
    scheme_names = schemes_df['Scheme'].fillna('').tolist()
    scheme_embeddings = processor.semantic_model.encode(scheme_names, convert_to_tensor=True)
    
    # Calculate similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1), 
        scheme_embeddings
    )[0]
    
    # Find top matches (schemes with similarity above threshold)
    similarity_threshold = 0.3  # Adjust based on testing
    relevant_indices = np.where(similarities > similarity_threshold)[0]
    
    # Sort by similarity score (descending)
    relevant_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    
    # Limit to top N matches
    top_n = 10
    relevant_indices = relevant_indices[:top_n]
    
    logger.info(f"Found {len(relevant_indices)} semantically relevant schemes")
    
    # Prepare the response
    schemes_list = []
    for idx in relevant_indices:
        row = schemes_df.iloc[idx]
        scheme_data = {
            'scheme_name': row['Scheme'],
            'ministry_department': row['Ministry/Department'],
            'budget_estimates_2021_2022': row['Budget Estimates2021-2022 - Total'],
            'actuals_2019_2020': row['Actuals2019-2020 - Total'],
            'similarity_score': float(similarities[idx])
        }
        schemes_list.append(scheme_data)
    
    # If no relevant schemes found by semantic search, try the original method
    if not schemes_list:
        logger.info("No schemes found by semantic search, falling back to code matching")
        return get_relevant_schemes_by_code(nic_code)
    
    return schemes_list

def get_relevant_schemes_by_code(nic_code):
    """Original method to fetch relevant government schemes based on NIC code"""
    schemes_df = get_govt_schemes()
    if schemes_df is None:
        return []
    
    # Convert NIC code to string for comparison
    nic_code_str = str(nic_code)
    
    # Filter schemes where the NIC code is present in the 'Category' column
    relevant_schemes = schemes_df[
        schemes_df['Category'].apply(
            lambda x: nic_code_str in str(x).split(',') if pd.notna(x) else False
        )
    ]
    
    if relevant_schemes.empty:
        return []
    
    # Prepare the response
    schemes_list = []
    for _, row in relevant_schemes.iterrows():
        scheme_data = {
            'scheme_name': row['Scheme'],
            'ministry_department': row['Ministry/Department'],
            'budget_estimates_2021_2022': row['Budget Estimates2021-2022 - Total'],
            'actuals_2019_2020': row['Actuals2019-2020 - Total']
        }
        schemes_list.append(scheme_data)
    
    return schemes_list

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Load and process image
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Run image analysis
        predictions = predict_items(image_data)
        
        if not predictions:
            return jsonify({'error': 'Could not process image or no predictions found'}), 400
        
        # Generate a description from the predicted items
        item_descriptions = [f"{label} ({prob*100:.1f}%)" for label, prob in predictions]
        business_description = "Business dealing with " + ", ".join(item_descriptions[:3])
        
        return jsonify({
            'predictions': [{'label': label, 'probability': prob} for label, prob in predictions],
            'suggested_description': business_description
        })
    
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_nic_codes', methods=['POST'])
def get_nic_codes():
    try:
        # Lazy initialization of processor
        processor = get_processor()
        if processor is None:
            return jsonify({
                'error': 'NIC Code Processor failed to initialize. Check server logs.',
                'results': []
            }), 500
        
        # Get input
        user_input = request.json.get('input', '')
        original_input = request.json.get('original_input', user_input)
        
        logger.info(f"Received request for: {user_input}")
        logger.info(f"Original input was: {original_input}")
        
        # Find NIC codes
        result_data = processor.find_nic_codes(user_input)
        
        # Add relevant schemes to each NIC code result
        for result in result_data['results']:
            nic_code = result['nic_code']
            relevant_schemes = get_relevant_schemes(nic_code)
            result['relevant_schemes'] = relevant_schemes
        
        return jsonify({
            'results': result_data['results'],
            'total_matches': result_data['total_matches'],
            'original_input': original_input,
            'enhanced_input': user_input if user_input != original_input else None,
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'processor_initialized': get_processor() is not None,
        'image_model_initialized': get_image_model() is not None,
        'govt_schemes_loaded': get_govt_schemes() is not None
    })

@app.route('/batch_process', methods=['POST'])
def batch_process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is a text file
        if not file.filename.endswith(('.txt', '.csv')):
            return jsonify({'error': 'Only .txt and .csv files are supported'}), 400
            
        # Process the file content
        content = file.read().decode('utf-8')
        
        # Split content by new lines, filtering out empty lines
        descriptions = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Get processor
        processor = get_processor()
        if processor is None:
            return jsonify({
                'error': 'NIC Code Processor failed to initialize. Check server logs.',
                'results': []
            }), 500
        
        # Process each description
        batch_results = []
        for desc in descriptions:
            # Find NIC codes for this description
            result_data = processor.find_nic_codes(desc)
            
            # Add relevant schemes to each NIC code result
            for result in result_data['results']:
                nic_code = result['nic_code']
                relevant_schemes = get_relevant_schemes(nic_code)
                result['relevant_schemes'] = relevant_schemes
            
            batch_results.append({
                'description': desc,
                'results': result_data['results'],
                'total_matches': result_data['total_matches']
            })
        
        return jsonify({
            'batch_results': batch_results,
            'total_processed': len(batch_results)
        })
    
    except Exception as e:
        logger.error(f"Error processing batch file: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/feedback', methods=['POST'])
def collect_feedback():
    """Collect user feedback on NIC code matches"""
    try:
        feedback_data = request.json
        logger.info(f"Received feedback: {feedback_data}")
        
        # Placeholder for feedback collection logic
        return jsonify({'status': 'feedback received'})
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_relevant_schemes', methods=['GET'])
def get_relevant_schemes_endpoint():
    try:
        # Get NIC code from query parameters
        nic_code = request.args.get('nic_code')
        description = request.args.get('description')
        
        if not nic_code:
            return jsonify({'error': 'NIC code is required'}), 400
        
        # Fetch relevant schemes with semantic similarity if description is provided
        schemes = get_relevant_schemes(nic_code, description)
        
        # Ensure the response is always an array
        if not schemes:
            return jsonify([])
        
        return jsonify(schemes)
    except Exception as e:
        logger.error(f"Error fetching government schemes: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    
    # Pre-initialize processor in a background thread
    import threading
    threading.Thread(target=get_processor, daemon=True).start()
    threading.Thread(target=get_image_model, daemon=True).start()
    threading.Thread(target=get_govt_schemes, daemon=True).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

