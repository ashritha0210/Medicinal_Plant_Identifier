import os
from urllib.parse import quote_plus
from flask import Flask, request, render_template, url_for, jsonify, session, redirect, flash
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from datetime import timedelta, datetime
from functools import wraps
from dotenv import load_dotenv
import certifi
import numpy as np

# Configure TensorFlow for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.permanent_session_lifetime = timedelta(days=1)

# Load environment variables
load_dotenv()

# Database Connection with error handling
try:
    client = MongoClient(
        f"mongodb+srv://{quote_plus(os.getenv('DB_USER'))}:{quote_plus(os.getenv('DB_PASSWORD'))}@"
        f"{os.getenv('DB_CLUSTER')}/?retryWrites=true&w=majority",
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
    # Test the connection
    client.admin.command('ping')
    db = client['plant_auth_db']
    users_collection = db['users']
    print("✅ Successfully connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    raise

# Universal Model Loader for CPU/GPU
def load_universal_model(model_path):
    # Detect available devices and configure accordingly
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"🚀 Using GPU: {gpus[0].name}")
        # Enable memory growth to prevent TF from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Load model with GPU support
        model = load_model(model_path)
    else:
        print("💻 Using CPU")
        # Explicitly load model on CPU
        with tf.device('/CPU:0'):
            model = load_model(model_path)
    
    # Warm up the model
    dummy_input = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    return model

# Load the plant identification model
MODEL_PATH = '/content/drive/MyDrive/Mini_pro/planteffnet.keras'
try:
    model = load_universal_model(MODEL_PATH)
    labels = sorted(os.listdir('/content/drive/MyDrive/Medicinal Leaf dataset'))
    print("✅ Plant identification model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise
# Plant information database (your existing PLANT_INFO dictionary remains the same)
PLANT_INFO = {
    'Aloevera': {
        'medicinal_properties': 'Used for skin healing, digestion, and boosting immunity.',
        'sustainability': 'Aloe vera waste can be composted or used to make natural fertilizers.'
    },
    'Amla': {
        'medicinal_properties': 'Rich in vitamin C, improves immunity, and promotes hair growth.',
        'sustainability': 'Amla seeds and pulp can be used to make natural dyes or composted.'
    },
    'Amruthaballi': {
        'medicinal_properties': 'Known for its anti-inflammatory and immune-boosting properties.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Arali': {
        'medicinal_properties': 'Used in traditional medicine for treating skin diseases and fever.',
        'sustainability': 'Arali leaves can be composted or used as natural pesticides.'
    },
    'Astma_weed': {
        'medicinal_properties': 'Helps in treating respiratory disorders like asthma and bronchitis.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Badipala': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The leaves can be composted or used as natural fertilizers.'
    },
    'Balloon_Vine': {
        'medicinal_properties': 'Known for its anti-inflammatory and pain-relieving properties.',
        'sustainability': 'The plant can be composted or used as mulch.'
    },
    'Bamboo': {
        'medicinal_properties': 'Used for treating coughs and improving digestion.',
        'sustainability': 'Bamboo waste can be used for making crafts, furniture, or composted.'
    },
    'Beans': {
        'medicinal_properties': 'Rich in protein and fiber, promotes heart health.',
        'sustainability': 'Bean pods can be composted or used as animal feed.'
    },
    'Betel': {
        'medicinal_properties': 'Used for improving digestion and treating respiratory issues.',
        'sustainability': 'Betel leaves can be composted or used as natural pesticides.'
    },
    'Bhrami': {
        'medicinal_properties': 'Improves memory, reduces anxiety, and promotes hair growth.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Bringaraja': {
        'medicinal_properties': 'Used for treating skin diseases and promoting hair growth.',
        'sustainability': 'The leaves can be composted or used as natural fertilizers.'
    },
    'Caricature': {
        'medicinal_properties': 'Used for treating skin infections and inflammation.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Castor': {
        'medicinal_properties': 'Used for treating constipation and skin disorders.',
        'sustainability': 'Castor oil waste can be used as a natural lubricant or composted.'
    },
    'Catharanthus': {
        'medicinal_properties': 'Used in cancer treatment and for managing diabetes.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Chakte': {
        'medicinal_properties': 'Used for treating skin diseases and digestive disorders.',
        'sustainability': 'The bark and leaves can be composted or used as natural dyes.'
    },
    'Chilly': {
        'medicinal_properties': 'Rich in capsaicin, helps in pain relief and boosting metabolism.',
        'sustainability': 'Chilli waste can be composted or used as natural pesticides.'
    },
    'Citron lime (herelikai)': {
        'medicinal_properties': 'Rich in vitamin C, improves immunity and digestion.',
        'sustainability': 'Citron lime peels can be composted or used as natural cleaners.'
    },
    'Coffee': {
        'medicinal_properties': 'Improves focus and energy levels.',
        'sustainability': 'Coffee grounds can be composted or used as a natural fertilizer.'
    },
    'Common rue(naagdalli)': {
        'medicinal_properties': 'Used for treating digestive disorders and improving eyesight.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Coriender': {
        'medicinal_properties': 'Improves digestion and reduces inflammation.',
        'sustainability': 'Coriander waste can be composted or used as natural pesticides.'
    },
    'Curry': {
        'medicinal_properties': 'Improves digestion and has anti-inflammatory properties.',
        'sustainability': 'Curry leaves can be composted or used as natural fertilizers.'
    },
    'Doddpathre': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The leaves can be composted or used as mulch.'
    },
    'Drumstick': {
        'medicinal_properties': 'Rich in nutrients, improves immunity and bone health.',
        'sustainability': 'Drumstick pods and leaves can be composted or used as animal feed.'
    },
    'Ekka': {
        'medicinal_properties': 'Used for treating respiratory disorders and fever.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Eucalyptus': {
        'medicinal_properties': 'Used for treating respiratory issues and as an antiseptic.',
        'sustainability': 'Eucalyptus leaves can be composted or used as natural insecticides.'
    },
    'Ganigale': {
        'medicinal_properties': 'Used for treating skin diseases and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Ganike': {
        'medicinal_properties': 'Used for treating fever and respiratory disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Gasagase': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The seeds can be composted or used as natural fertilizers.'
    },
    'Ginger': {
        'medicinal_properties': 'Improves digestion, reduces nausea, and has anti-inflammatory properties.',
        'sustainability': 'Ginger waste can be composted or used as natural cleaners.'
    },
    'Globe Amarnath': {
        'medicinal_properties': 'Rich in antioxidants, improves immunity and digestion.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Guava': {
        'medicinal_properties': 'Rich in vitamin C, improves immunity and digestion.',
        'sustainability': 'Guava leaves and peels can be composted or used as natural fertilizers.'
    },
    'Henna': {
        'medicinal_properties': 'Used for skin cooling, hair conditioning, and wound healing.',
        'sustainability': 'Henna leaves can be dried and powdered for reuse, and the waste can be composted.'
    },
    'Hibiscus': {
        'medicinal_properties': 'Improves hair health, reduces blood pressure, and boosts immunity.',
        'sustainability': 'Hibiscus flowers and leaves can be composted or used as natural dyes.'
    },
    'Honge': {
        'medicinal_properties': 'Used for treating skin diseases and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as natural pesticides.'
    },
    'Insulin': {
        'medicinal_properties': 'Helps in managing diabetes and improving digestion.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Jackfruit': {
        'medicinal_properties': 'Rich in nutrients, improves digestion and immunity.',
        'sustainability': 'Jackfruit peels and seeds can be composted or used as animal feed.'
    },
    'Jasmine': {
        'medicinal_properties': 'Used for reducing stress and improving skin health.',
        'sustainability': 'Jasmine flowers can be composted or used as natural perfumes.'
    },
    'Kambajala': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Kasambruga': {
        'medicinal_properties': 'Used for treating fever and respiratory disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Kohlrabi': {
        'medicinal_properties': 'Rich in vitamins, improves digestion and immunity.',
        'sustainability': 'Kohlrabi leaves can be composted or used as animal feed.'
    },
    'Lantana': {
        'medicinal_properties': 'Used for treating skin infections and inflammation.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Lemon': {
        'medicinal_properties': 'Rich in vitamin C, improves immunity and digestion.',
        'sustainability': 'Lemon peels can be composted or used as natural cleaners.'
    },
    'Lemongrass': {
        'medicinal_properties': 'Improves digestion, reduces stress, and has anti-inflammatory properties.',
        'sustainability': 'Lemongrass waste can be composted or used as natural insecticides.'
    },
    'Malabar_Nut': {
        'medicinal_properties': 'Used for treating respiratory disorders and fever.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Malabar_Spinach': {
        'medicinal_properties': 'Rich in nutrients, improves digestion and immunity.',
        'sustainability': 'The leaves can be composted or used as natural fertilizers.'
    },
    'Mango': {
        'medicinal_properties': 'Rich in vitamins A and C, boosts immunity, and aids digestion.',
        'sustainability': 'Mango peels and seeds can be composted or used to make natural dyes.'
    },
    'Marigold': {
        'medicinal_properties': 'Used for treating skin infections and improving eyesight.',
        'sustainability': 'Marigold flowers can be composted or used as natural dyes.'
    },
    'Mint': {
        'medicinal_properties': 'Helps with digestion, relieves headaches, and has anti-inflammatory properties.',
        'sustainability': 'Mint waste can be used to make natural insect repellents or composted.'
    },
    'Neem': {
        'medicinal_properties': 'Used for treating skin disorders, improving liver health, and boosting immunity.',
        'sustainability': 'Neem leaves and bark can be used as natural pesticides, and the waste can be composted.'
    },
    'Nelavembu': {
        'medicinal_properties': 'Used for treating fever and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Nerale': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Nooni': {
        'medicinal_properties': 'Used for treating skin diseases and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Onion': {
        'medicinal_properties': 'Rich in antioxidants, improves heart health and immunity.',
        'sustainability': 'Onion peels can be composted or used as natural dyes.'
    },
    'Padri': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Palak(Spinach)': {
        'medicinal_properties': 'Rich in iron and vitamins, improves immunity and digestion.',
        'sustainability': 'Spinach waste can be composted or used as animal feed.'
    },
    'Papaya': {
        'medicinal_properties': 'Improves digestion, boosts immunity, and promotes skin health.',
        'sustainability': 'Papaya peels and seeds can be composted or used as natural fertilizers.'
    },
    'Parijatha': {
        'medicinal_properties': 'Used for treating fever and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Pea': {
        'medicinal_properties': 'Rich in protein and fiber, promotes heart health.',
        'sustainability': 'Pea pods can be composted or used as animal feed.'
    },
    'Pepper': {
        'medicinal_properties': 'Improves digestion and has anti-inflammatory properties.',
        'sustainability': 'Pepper waste can be composted or used as natural pesticides.'
    },
    'Pomoegranate': {
        'medicinal_properties': 'Rich in antioxidants, improves heart health and immunity.',
        'sustainability': 'Pomegranate peels can be composted or used as natural dyes.'
    },
    'Pumpkin': {
        'medicinal_properties': 'Rich in vitamins, improves digestion and immunity.',
        'sustainability': 'Pumpkin waste can be composted or used as animal feed.'
    },
    'Raddish': {
        'medicinal_properties': 'Improves digestion and boosts immunity.',
        'sustainability': 'Radish leaves can be composted or used as animal feed.'
    },
    'Rose': {
        'medicinal_properties': 'Used for reducing stress and improving skin health.',
        'sustainability': 'Rose petals can be composted or used as natural perfumes.'
    },
    'Sampige': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Sapota': {
        'medicinal_properties': 'Rich in nutrients, improves digestion and immunity.',
        'sustainability': 'Sapota peels and seeds can be composted or used as natural fertilizers.'
    },
    'Seethaashoka': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Seethapala': {
        'medicinal_properties': 'Used for treating fever and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Spinach1': {
        'medicinal_properties': 'Rich in iron and vitamins, improves immunity and digestion.',
        'sustainability': 'Spinach waste can be composted or used as animal feed.'
    },
    'Tamarind': {
        'medicinal_properties': 'Improves digestion and has anti-inflammatory properties.',
        'sustainability': 'Tamarind seeds and pulp can be composted or used as natural dyes.'
    },
    'Taro': {
        'medicinal_properties': 'Rich in nutrients, improves digestion and immunity.',
        'sustainability': 'Taro leaves and stems can be composted or used as animal feed.'
    },
    'Tecoma': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Thumbe': {
        'medicinal_properties': 'Used for treating fever and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'Tomato': {
        'medicinal_properties': 'Rich in antioxidants, improves heart health and immunity.',
        'sustainability': 'Tomato waste can be composted or used as natural fertilizers.'
    },
    'Tulsi': {
        'medicinal_properties': 'Known for its anti-inflammatory, antioxidant, and immune-boosting properties.',
        'sustainability': 'Tulsi leaves can be dried and used as herbal tea, and the plant waste can be composted.'
    },
    'Turmeric': {
        'medicinal_properties': 'Anti-inflammatory, antioxidant, and aids in digestion.',
        'sustainability': 'Turmeric waste can be composted or used as a natural dye.'
    },
    'ashoka': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'camphor': {
        'medicinal_properties': 'Used for treating respiratory disorders and as an antiseptic.',
        'sustainability': 'Camphor waste can be composted or used as natural insecticides.'
    },
    'kamakasturi': {
        'medicinal_properties': 'Used for treating digestive disorders and skin infections.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    },
    'kepala': {
        'medicinal_properties': 'Used for treating fever and digestive disorders.',
        'sustainability': 'The plant waste can be composted or used as mulch.'
    }

}

# For plants not in our database, provide default information
DEFAULT_INFO = {
    "medicinal_properties": "This plant has various traditional medicinal uses. Further research is recommended.",
    "sustainability": "Plant matter can be composted or used as mulch. Check local guidelines for specific decomposition practices."
}
# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
@login_required
def home():
    return render_template('index.html', username=session['user']['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user'] = {
                'username': user['username'],
                'email': user['email'],
                'last_login': datetime.utcnow()
            }
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
        elif users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            flash('Username or email already exists', 'danger')
        else:
            users_collection.insert_one({
                'username': username,
                'email': email,
                'password': generate_password_hash(password),
                'created_at': datetime.utcnow(),
                'last_login': datetime.utcnow()
            })
            flash('Account created successfully! Please log in', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Create upload directory if it doesn't exist
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # Process image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        #img_array = preprocess_input(img_array)
        img_array = img_array.astype(np.float32)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        
        # Get plant info
        plant_info = PLANT_INFO.get(predicted_class, DEFAULT_INFO)
        
        return jsonify({
            'success': True,
            'result': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'image_url': url_for('static', filename=f'uploads/{file.filename}'),
            'medicinal_properties': plant_info['medicinal_properties'],
            'sustainability': plant_info['sustainability']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Start Ngrok tunnel if needed (for development)
def start_ngrok():
    try:
        from pyngrok import ngrok
        # Get your ngrok auth token from https://dashboard.ngrok.com/get-started/your-authtoken
        ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN'))  
        public_url = ngrok.connect(5000).public_url
        print(f" * Ngrok tunnel running at: {public_url}")
    except ImportError:
        print(" * Ngrok not available, running locally only")
    except Exception as e:
        print(f" * Ngrok error: {e}")

if __name__ == '__main__':
    # Create upload directory
    os.makedirs('static/uploads', exist_ok=True)
    
    # Get the local IP address
    from socket import gethostbyname, gethostname
    local_ip = gethostbyname(gethostname())
    
    print(f"\n🌿 Access your application at:")
    print(f"• Local: http://localhost:5000")
    print(f"• Network: http://{local_ip}:5000")
    
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000).public_url
        print(f"• Public: {public_url}")
    except:
        print("• Ngrok not configured (public access unavailable)")
    
    # Run with explicit host and port
    app.run(host='0.0.0.0', port=5000, debug=False)
