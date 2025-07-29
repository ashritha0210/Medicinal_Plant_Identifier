import os
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from urllib.parse import quote_plus
import certifi
from datetime import datetime

# Load environment variables
load_dotenv()

def initialize_database():
    client = None
    try:
        # Verify critical environment variables
        required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_CLUSTER']
        if not all(os.getenv(var) for var in required_vars):
            missing = [var for var in required_vars if not os.getenv(var)]
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

        # Get credentials
        username = os.getenv("DB_USER")  # ashritha22241a6722
        password = os.getenv("DB_PASSWORD")  # Ashritha2
        cluster = os.getenv("DB_CLUSTER")  # plantdetection.kdkfhek.mongodb.net

        # Build connection string (no special chars need encoding now)
        connection_string = (
            f"mongodb+srv://{username}:{password}@{cluster}/"
            "admin?retryWrites=true&w=majority&"
            "tls=true&authSource=admin"
        )

        # Connect with SSL verification
        client = MongoClient(
            connection_string,
            tlsCAFile=certifi.where(),
            connectTimeoutMS=10000,
            socketTimeoutMS=30000
        )

        # Test connection
        print("🔄 Testing MongoDB connection...")
        client.admin.command('ping')
        print("✅ Connection successful")

        # Initialize database
        db = client.plant_auth_db
        users = db.users

        # Create admin user if not exists
        admin_password = os.getenv('ADMIN_INIT_PASSWORD', 'admin123')
        result = users.update_one(
            {'username': 'admin'},
            {'$setOnInsert': {
                'username': 'admin',
                'email': 'admin@example.com',
                'password': generate_password_hash(admin_password),
                'role': 'admin',
                'createdAt': datetime.utcnow()
            }},
            upsert=True
        )

        if result.upserted_id:
            print(f"👤 Admin user created with password: {admin_password}")
            print("⚠️ Change this password immediately after first login!")
        else:
            print("ℹ️ Admin user already exists")

        # Create indexes
        users.create_index([('username', 1)], unique=True)
        users.create_index([('email', 1)], unique=True)
        print("📊 Database indexes created")

        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        if hasattr(e, 'details'):
            print(f"MongoDB details: {e.details}")
        return False
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌿 Initializing Medicinal Plant Database")
    print("="*50)
    
    if initialize_database():
        print("\n✅ Setup completed successfully!")
        exit(0)
    else:
        print("\n❌ Setup failed. Check errors above.")
        exit(1)