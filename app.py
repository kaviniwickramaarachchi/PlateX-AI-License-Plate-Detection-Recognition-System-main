import os
import cv2
import csv
from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import re

# Import numpy with error handling
try:
    import numpy as np
    print("NumPy imported successfully")
except ImportError as e:
    print(f"NumPy import error: {e}")
    print("Please install numpy: pip install numpy")
    exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['CSV_FILE'] = 'vehicle_data.csv'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize CSV file with headers if it doesn't exist
def initialize_csv():
    csv_path = app.config['CSV_FILE']
    if not os.path.exists(csv_path):
        headers = [
            'Timestamp',
            'License_Plate',
            'Confidence',
            'Province_Name',
            'Province_Code',
            'Vehicle_Category',
            'Fuel_Type',
            'Image_Filename'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        print(f"CSV file created: {csv_path}")

# Initialize CSV file on startup
initialize_csv()

def append_to_csv(vehicle_data):
    """Append vehicle detection data to CSV file"""
    try:
        csv_path = app.config['CSV_FILE']
        
        # Extract data from vehicle_data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        license_plate = vehicle_data.get('formatted_text', vehicle_data.get('text', ''))
        confidence = vehicle_data.get('confidence', 0)
        
        # Extract vehicle details
        details = vehicle_data.get('vehicle_details', {})
        province_name = details.get('province', {}).get('name', '') if details.get('province') else ''
        province_code = details.get('province', {}).get('code', '') if details.get('province') else ''
        vehicle_category = details.get('vehicle_category', '')
        fuel_type = details.get('fuel_type', '')
        
        # Image filename (if available)
        image_filename = vehicle_data.get('image_filename', '')
        
        # Prepare row data
        row_data = [
            timestamp,
            license_plate,
            f"{confidence:.3f}",
            province_name,
            province_code,
            vehicle_category,
            fuel_type,
            image_filename
        ]
        
        # Append to CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        
        print(f"Data appended to CSV: {license_plate}")
        return True
        
    except Exception as e:
        print(f"Error appending to CSV: {str(e)}")
        return False

# Initialize YOLO model and OCR
print("Loading YOLO model...")
try:
    model = YOLO('best.pt')
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

print("Loading OCR model...")
try:
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    print("OCR model loaded successfully!")
except Exception as e:
    print(f"Error loading OCR model: {e}")
    exit(1)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_detected_plate(detected_text):
    """
    Format detected license plate text into proper Sri Lankan format
    
    Examples:
    - "EPCAS5031" → "EP CAS 5031"
    - "WP181234" → "WP 18 1234"  
    - "SGPB9881" → "SG PB 9881"
    - "561234" → "56 1234"
    - "3251234" → "325 1234"
    - "E1234" → "E? 1234" (single letter codes like E,G,K,H,J,Q,T,M,V,B,P need second letter)
    - "A1234" → "A?? 1234" (codes starting with A,B,C,D need 3 letters like AAA,BAC,CAS,DAG)
    """
    
    # Remove all spaces and hyphens first, keep only alphanumeric
    clean_text = ''.join(c for c in detected_text.upper() if c.isalnum())
    
    if not clean_text:
        return detected_text
    
    
    # Province codes for identification
    PROVINCE_CODES = ['WP', 'SP', 'EP', 'NP', 'NC', 'CP', 'NW', 'SG', 'UP']
    
    # Try different formatting patterns based on content
    
    # Pattern 0: Check for province code at the end FIRST (e.g., "PB-9881 SG" → "SG PB 9881", "56-2543 SG" → "SG 56 2543")
    # Split original text to preserve spaces, then clean each part
    original_words = detected_text.upper().strip().split()
    if len(original_words) >= 2:
        last_word = original_words[-1]
        if last_word in PROVINCE_CODES:
            # Province at the end, move it to the front
            remaining_parts = original_words[:-1]
            remaining_text = ''.join(c for part in remaining_parts for c in part if c.isalnum())
            
            # Case 1: Letters + Numbers (like PB9881 → SG PB 9881)
            if len(remaining_text) >= 6 and not remaining_text.isdigit():  # At least 2 letters + 4 digits
                # Find where numbers start (look for 4 consecutive digits at the end)
                for i in range(len(remaining_text) - 3):
                    if remaining_text[i:i+4].isdigit() and i+4 == len(remaining_text):
                        letters = remaining_text[:i]
                        numbers = remaining_text[i:i+4]
                        if letters:
                            formatted = f"{last_word} {letters} {numbers}"
                            return formatted
                        break
            
            # Case 2: Pure Numbers (like 562543 → SG 56 2543)
            if remaining_text.isdigit():
                if len(remaining_text) == 6:  # NNNNNN
                    formatted = f"{last_word} {remaining_text[:2]} {remaining_text[2:]}"
                    return formatted
                elif len(remaining_text) == 7:  # NNNNNNN
                    formatted = f"{last_word} {remaining_text[:3]} {remaining_text[3:]}"
                    return formatted
    
    # Pattern 1: Check if starts with province code
    for province in PROVINCE_CODES:
        if clean_text.startswith(province):
            remaining = clean_text[2:]  # Remove province code
            
            # Check for single letter after province (e.g., WPE1234, WPA1234)
            if len(remaining) == 5 and remaining[1:].isdigit():  # Single letter + 4 digits
                first_char = remaining[0]
                two_letter_codes = ['E', 'G', 'K', 'H', 'J', 'Q', 'T', 'M', 'V', 'B', 'P']
                three_letter_codes = ['A', 'B', 'C', 'D']
                
                if first_char in two_letter_codes:
                    formatted = f"{province} {first_char}? {remaining[1:]}"
                    return formatted
                elif first_char in three_letter_codes:
                    formatted = f"{province} {first_char}?? {remaining[1:]}"
                    return formatted
            
            # Check what comes after province
            if len(remaining) >= 5:  # At least 1 letter + 4 digits
                # Find where numbers start
                letter_part = ''
                number_part = ''
                
                for i, char in enumerate(remaining):
                    if char.isdigit():
                        letter_part = remaining[:i]
                        number_part = remaining[i:]
                        break
                
                if letter_part and number_part and len(number_part) == 4:
                    formatted = f"{province} {letter_part} {number_part}"
                    return formatted
            
            # If remaining looks like numeric (e.g., WP181234)
            if len(remaining) >= 5 and remaining[-4:].isdigit():
                first_part = remaining[:-4]
                last_part = remaining[-4:]
                if first_part:
                    formatted = f"{province} {first_part} {last_part}"
                    return formatted
    
    # Pattern 2: Check if it's pure numeric (like 561234, 3251234)
    if clean_text.isdigit():
        if len(clean_text) == 6:  # NNNNNN
            # Could be NN NNNN or NNN NNN, try NN NNNN first
            formatted = f"{clean_text[:2]} {clean_text[2:]}"
            return formatted
        elif len(clean_text) == 7:  # NNNNNNN  
            # Try NNN NNNN
            formatted = f"{clean_text[:3]} {clean_text[3:]}"
            return formatted
    
    # Pattern 3: Handle single letter cases first (E1234, A1234, etc.)
    if not clean_text.isdigit() and len(clean_text) == 5:  # Single letter + 4 digits
        first_char = clean_text[0]
        if clean_text[1:].isdigit():  # Rest are digits
            # Single letter codes that need expansion
            two_letter_codes = ['E', 'G', 'K', 'H', 'J', 'Q', 'T', 'M', 'V', 'B', 'P']
            three_letter_codes = ['A', 'B', 'C', 'D']
            
            if first_char in two_letter_codes:
                formatted = f"{first_char}? {clean_text[1:]}"
                return formatted
            elif first_char in three_letter_codes:
                formatted = f"{first_char}?? {clean_text[1:]}"
                return formatted
    
    # Pattern 4: Mixed alpha-numeric (like SGPB9881, EPCAS5031, GI8712)
    if not clean_text.isdigit():
        # Find where the numbers start (usually last 4 digits)
        if len(clean_text) >= 4:
            # Look for last 4 consecutive digits
            for i in range(len(clean_text) - 3):
                if clean_text[i:i+4].isdigit() and (i+4 == len(clean_text) or not clean_text[i+4].isdigit()):
                    letters = clean_text[:i]
                    numbers = clean_text[i:i+4]
                    
                    if len(letters) >= 2:
                        # Check if first 2 letters are province
                        potential_province = letters[:2]
                        if potential_province in PROVINCE_CODES:
                            remaining_letters = letters[2:]
                            if remaining_letters:
                                formatted = f"{potential_province} {remaining_letters} {numbers}"
                                return formatted
                        else:
                            # No province, could be format like SGPB9881 → SG PB 9881, or GI8712 → GI 8712
                            if len(letters) == 4:  # Like SGPB
                                formatted = f"{letters[:2]} {letters[2:]} {numbers}"
                                return formatted
                            elif len(letters) >= 3:  # Like EPCAS → EP CAS
                                # Try to split reasonably
                                if len(letters) == 5:  # EPCAS
                                    formatted = f"{letters[:2]} {letters[2:]} {numbers}"
                                    return formatted
                                else:
                                    formatted = f"{letters} {numbers}"
                                    return formatted
                            elif len(letters) == 2:  # Like GI8712 → GI 8712
                                formatted = f"{letters} {numbers}"
                                return formatted
                    elif len(letters) >= 1:  # Handle cases like single letter + numbers
                        formatted = f"{letters} {numbers}"
                        return formatted
                    break
    
    # Pattern 5: Already has some separators, just clean up
    if any(c in detected_text for c in [' ', '-']):
        # Just normalize existing separators
        normalized = re.sub(r'[-\s]+', ' ', detected_text.upper().strip())
        return normalized
    
    # Fallback: return as-is if no pattern matches
    return detected_text

def parsePlate(input_plate):
    """
    Parse Sri Lankan license plate and extract vehicle information
    
    Args:
        input_plate (str): Raw license plate string
    
    Returns:
        dict: Parsed plate information including vehicle category, fuel type, and province
    """
    
    # Province mappings
    PROVINCES = {
        'WP': 'Western Province',
        'SP': 'Southern Province', 
        'EP': 'Eastern Province',
        'NP': 'Northern Province',
        'NC': 'North Central Province',
        'CP': 'Central Province',
        'NW': 'North Western Province',
        'SG': 'Sabaragamuwa Province',
        'UP': 'Uva Province'
    }
    
    # Normalize input
    if not input_plate:
        return {
            "plate_raw": input_plate,
            "plate_normalized": "",
            "format_version": "unknown",
            "province": None,
            "first_part": "",
            "numeric_part": "",
            "vehicle_category": "unknown",
            "fuel_type": "unknown",
            "category_code": "UNKNOWN",
            "notes": "Empty input"
        }
    
    # Clean and normalize
    normalized = input_plate.upper().strip()
    normalized = re.sub(r'[-\s]+', ' ', normalized)  # Replace multiple spaces/hyphens with single space
    normalized = re.sub(r'\s+', ' ', normalized)     # Replace multiple spaces with single space
    
    # Initialize result
    result = {
        "plate_raw": input_plate,
        "plate_normalized": normalized,
        "format_version": "unknown",
        "province": None,
        "first_part": "",
        "numeric_part": "",
        "vehicle_category": "unknown",
        "fuel_type": "unknown", 
        "category_code": "UNKNOWN",
        "notes": ""
    }
    
    # Format detection patterns
    patterns = {
        'v1': r'^(\d{1,2})\s+(\d{4})$',           # NN-NNNN or NN NNNN (e.g., 18-1234)
        'v2': r'^(\d{3})\s+(\d{4})$',             # NNN-NNNN or NNN NNNN (e.g., 325-1234)
        'v3': r'^([A-Z]{1,2})\s+(\d{4})$',        # L NNNN or LL NNNN (e.g., E 1234, KA 1234)
        'v4': r'^([A-Z]{2})\s+([A-Z]{1,2}|\d{1,2})\s+(\d{4})$',  # PP L NNNN or PP LL NNNN or PP NN NNNN (e.g., WP E 1234, WP KA 1234, WP 18 1234)
        'v5': r'^([A-Z]{3})\s+(\d{4})$',          # LLL NNNN (e.g., CAA 1234)
        'v6': r'^([A-Z]{2})\s+([A-Z]{3})\s+(\d{4})$'   # PP LLL NNNN (e.g., WP CAA 1234)
    }
    
    # Try to match patterns
    matched = False
    for version, pattern in patterns.items():
        match = re.match(pattern, normalized)
        if match:
            result["format_version"] = version
            groups = match.groups()
            
            if version == 'v1':  # NN-NNNN
                result["first_part"] = groups[0]
                result["numeric_part"] = groups[1]
            elif version == 'v2':  # NNN-NNNN
                result["first_part"] = groups[0]
                result["numeric_part"] = groups[1]
            elif version == 'v3':  # LL NNNN
                result["first_part"] = groups[0]
                result["numeric_part"] = groups[1]
            elif version == 'v4':  # PP LL NNNN or PP NN NNNN
                province_code = groups[0]
                if province_code in PROVINCES:
                    result["province"] = {"code": province_code, "name": PROVINCES[province_code]}
                result["first_part"] = groups[1]
                result["numeric_part"] = groups[2]
            elif version == 'v5':  # LLL NNNN
                result["first_part"] = groups[0]
                result["numeric_part"] = groups[1]
            elif version == 'v6':  # PP LLL NNNN
                province_code = groups[0]
                if province_code in PROVINCES:
                    result["province"] = {"code": province_code, "name": PROVINCES[province_code]}
                result["first_part"] = groups[1]
                result["numeric_part"] = groups[2]
            
            matched = True
            break
    
    if not matched:
        result["notes"] = "Unknown plate format"
        return result
    
    # Determine vehicle category and fuel type based on first_part
    first_part = result["first_part"]
    
    # Check if first_part is numeric
    if first_part.isdigit():
        num = int(first_part)
        
        # Numeric range mappings (exact matches first, then ranges)
        # Handle exact matches first
        if num == 20:
            result["vehicle_category"] = "Van"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_VAN"
        elif num == 39:
            result["vehicle_category"] = "Heavy vehicle"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_HEAVY"
        elif num == 49:
            result["vehicle_category"] = "Heavy vehicle"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_HEAVY"
        elif num == 325:
            result["vehicle_category"] = "Converted vehicle"
            result["fuel_type"] = "any"
            result["category_code"] = "CONVERTED_VEHICLE"
        # Handle ranges (with priority for overlaps)
        elif 22 <= num <= 30:
            result["vehicle_category"] = "Heavy vehicle"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_HEAVY"
        elif 1 <= num <= 27 and num != 20:
            # Only if not already handled by diesel heavy range (22-30)
            if not (22 <= num <= 30):
                result["vehicle_category"] = "Motor car"
                result["fuel_type"] = "petrol"
                result["category_code"] = "PETROL_CAR"
        elif 34 <= num <= 38:
            result["vehicle_category"] = "Heavy vehicle"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_HEAVY"
        elif 40 <= num <= 48:
            result["vehicle_category"] = "Heavy vehicle"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_HEAVY"
        elif 50 <= num <= 59:
            result["vehicle_category"] = "Dual purpose"
            result["fuel_type"] = "any"
            result["category_code"] = "DUAL_PURPOSE"
        elif 60 <= num <= 63:
            result["vehicle_category"] = "Passenger vehicle"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_PASSENGER"
        elif 64 <= num <= 65:
            result["vehicle_category"] = "Motor car"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_CAR"
        elif 70 <= num <= 79:
            result["vehicle_category"] = "Tractor"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_TRACTOR"
        elif 80 <= num <= 160:
            result["vehicle_category"] = "Motorbike"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_MOTORBIKE"
        elif 200 <= num <= 208:
            result["vehicle_category"] = "Motor tricycle"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_TRICYCLE"
        elif 250 <= num <= 254:
            result["vehicle_category"] = "Light vehicle"
            result["fuel_type"] = "diesel"
            result["category_code"] = "DIESEL_LIGHT"
        elif 300 <= num <= 302:
            result["vehicle_category"] = "Motor car"
            result["fuel_type"] = "petrol"
            result["category_code"] = "PETROL_CAR"
        else:
            result["vehicle_category"] = "unknown"
            result["fuel_type"] = "unknown"
            result["category_code"] = "UNKNOWN_NUMERIC"
            result["notes"] = f"Unknown numeric code: {num}"
    
    else:
        # Alpha/mixed code mappings (exact matches have priority)
        alpha_mappings = {
            'E': ("Auction/process vehicle", "any", "AUCTION_VEHICLE"),
            'FZ': ("DBL/EPC vehicle", "any", "DBL_EPC_VEHICLE"),
            'G': ("All category vehicle", "any", "ALL_CATEGORY"),
            'H': ("All category vehicle", "any", "ALL_CATEGORY"),
            'J': ("All category vehicle", "any", "ALL_CATEGORY"),
            'K': ("Motor car", "any", "ANY_FUEL_CAR"),
            'LW': ("Ambulance", "any", "AMBULANCE"),
            'LY': ("Prime mover", "diesel", "DIESEL_PRIME_MOVER"),
            'LZ': ("Hertz/rental", "any", "RENTAL_VEHICLE"),
            'M': ("Motorbike", "petrol", "PETROL_MOTORBIKE"),
            'T': ("Motorbike", "petrol", "PETROL_MOTORBIKE"),
            'V': ("Motorbike", "petrol", "PETROL_MOTORBIKE"),
            'B': ("Motorbike", "petrol", "PETROL_MOTORBIKE"),
            'Q': ("Motor tricycle", "petrol", "PETROL_TRICYCLE"),
            'Y': ("Motor tricycle", "petrol", "PETROL_TRICYCLE"),
            'A': ("Motor tricycle", "petrol", "PETROL_TRICYCLE"),
            'BA': ("Dual purpose", "any", "DUAL_PURPOSE"),
            'TY': ("Dual purpose", "any", "DUAL_PURPOSE"),
            'RA': ("Tractor", "diesel", "DIESEL_TRACTOR"),
            'RR': ("Tractor", "diesel", "DIESEL_TRACTOR"),
            'RS': ("Land vehicle", "any", "LAND_VEHICLE"),
            'RU': ("Land vehicle", "any", "LAND_VEHICLE"),
            'RV': ("Tractor trailer", "diesel", "DIESEL_TRACTOR_TRAILER"),
            'RZ': ("Tractor trailer", "diesel", "DIESEL_TRACTOR_TRAILER"),
            'C': ("Motor car", "any", "MOTOR_CAR"),
            'P': ("Dual purpose", "any", "DUAL_PURPOSE"),
            'D': ("Dual purpose", "any", "DUAL_PURPOSE"),
            '98': ("DBL/EPC vehicle", "any", "DBL_EPC_VEHICLE"),
            '99': ("DBL/EPC vehicle", "any", "DBL_EPC_VEHICLE")
        }
        
        # Check exact matches first (highest priority)
        if first_part in alpha_mappings:
            category, fuel, code = alpha_mappings[first_part]
            result["vehicle_category"] = category
            result["fuel_type"] = fuel
            result["category_code"] = code
        
        # Check for codes starting with specific letters (like GI, GA, KA, EA, etc.) - lower priority
        if not first_part in alpha_mappings and len(first_part) >= 2:
            first_letter = first_part[0]
            
            if first_letter in ['G', 'H', 'J']:
                result["vehicle_category"] = "All category vehicle"
                result["fuel_type"] = "any"
                result["category_code"] = "ALL_CATEGORY"
                return result
            elif first_letter == 'K':
                result["vehicle_category"] = "Motor car"
                result["fuel_type"] = "any"
                result["category_code"] = "ANY_FUEL_CAR"
                return result
            elif first_letter == 'E':
                result["vehicle_category"] = "Auction/process vehicle"
                result["fuel_type"] = "any"
                result["category_code"] = "AUCTION_VEHICLE"
                return result
            elif first_letter in ['M', 'T', 'V']:
                result["vehicle_category"] = "Motorbike"
                result["fuel_type"] = "petrol"
                result["category_code"] = "PETROL_MOTORBIKE"
                return result
            elif first_letter == 'B' and len(first_part) == 2:
                # B codes as 2-letter (like BA) are handled by exact matches
                # Only 3-letter B codes (like BBB) should be motor tricycles
                pass
            elif first_letter in ['Q', 'Y']:
                result["vehicle_category"] = "Motor tricycle"
                result["fuel_type"] = "petrol"
                result["category_code"] = "PETROL_TRICYCLE"
                return result
            elif first_letter == 'P':
                result["vehicle_category"] = "Dual purpose"
                result["fuel_type"] = "any"
                result["category_code"] = "DUAL_PURPOSE"
                return result
            elif first_letter == 'A' and len(first_part) == 3:
                # A codes must be 3 letters (like AAA), not 2 letters (like AA)
                result["vehicle_category"] = "Motor tricycle"
                result["fuel_type"] = "petrol"
                result["category_code"] = "PETROL_TRICYCLE"
                return result
        
        # Check range patterns and special 3-letter codes
        if len(first_part) == 3:
            first_letter_3 = first_part[0]
            if first_letter_3 == 'A':
                # 3-letter codes starting with A (like AAA) are Motor tricycles (any fuel)
                result["vehicle_category"] = "Motor tricycle"
                result["fuel_type"] = "any"
                result["category_code"] = "ANY_FUEL_TRICYCLE"
            elif first_letter_3 == 'B':
                # 3-letter codes starting with B (like BBB) are Motorbikes
                result["vehicle_category"] = "Motorbike"
                result["fuel_type"] = "petrol"
                result["category_code"] = "PETROL_MOTORBIKE"
            elif first_letter_3 == 'C':
                # 3-letter codes starting with C (like CCC) are Motor cars
                result["vehicle_category"] = "Motor car"
                result["fuel_type"] = "any"
                result["category_code"] = "MOTOR_CAR"
            elif first_letter_3 == 'D':
                # 3-letter codes starting with D (like DDD) are Dual purpose
                result["vehicle_category"] = "Dual purpose"
                result["fuel_type"] = "any"
                result["category_code"] = "DUAL_PURPOSE"
            else:
                # Other 3-letter codes fall through to unknown
                result["vehicle_category"] = "unknown_alpha_code"
                result["fuel_type"] = "unknown"
                result["category_code"] = "UNKNOWN_ALPHA"
                result["notes"] = f"Unknown 3-letter code: {first_part}"
        elif first_part.startswith('L') and len(first_part) == 2:
            # LA-LY range for Lorry (catch remaining L codes)
            if 'LA' <= first_part <= 'LY' and first_part not in ['LW', 'LY', 'LZ']:
                result["vehicle_category"] = "Lorry"
                result["fuel_type"] = "diesel"
                result["category_code"] = "DIESEL_LORRY"
            else:
                result["vehicle_category"] = "unknown_alpha_code"
                result["fuel_type"] = "unknown"
                result["category_code"] = "UNKNOWN_ALPHA"
                result["notes"] = f"Unknown alpha code: {first_part}"
        elif first_part.startswith('N') and len(first_part) == 2:
            # NA-NZ range for Bus
            if 'NA' <= first_part <= 'NZ':
                result["vehicle_category"] = "Bus"
                result["fuel_type"] = "diesel"
                result["category_code"] = "DIESEL_BUS"
            else:
                result["vehicle_category"] = "unknown_alpha_code"
                result["fuel_type"] = "unknown"
                result["category_code"] = "UNKNOWN_ALPHA"
                result["notes"] = f"Unknown alpha code: {first_part}"
        else:
            result["vehicle_category"] = "unknown_alpha_code"
            result["fuel_type"] = "unknown"
            result["category_code"] = "UNKNOWN_ALPHA"
            result["notes"] = f"Unknown alpha code: {first_part}"
    
    return result

def detect_and_read_plate(image_path):
    """Detect license plates and extract text using OCR"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image file"}
        
        print(f"Processing image: {image.shape}")
        
        # Resize image for better processing
        image = cv2.resize(image, (1020, 600))
        
        # Run YOLO detection
        print("Running YOLO detection...")
        results = model(image)
        print(f"YOLO detection completed. Found {len(results)} results")
        
        detected_plates = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"Processing {len(boxes)} detections")
                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        print(f"Detection: class={model.names[class_id]}, confidence={confidence:.2f}")
                        
                        # Check if confidence is high enough and it's a license plate
                        if confidence > 0.5 and model.names[class_id].lower() == 'license_plate':
                            print("License plate detected, running OCR...")
                            
                            # Crop the license plate region
                            cropped_plate = image[y1:y2, x1:x2]
                            
                            if cropped_plate.size > 0:
                                # Run OCR on cropped plate
                                try:
                                    ocr_result = ocr.ocr(cropped_plate)
                                    
                                    # Extract text from OCR result
                                    plate_text = ""
                                    print(f"OCR raw result keys: {list(ocr_result[0].keys()) if ocr_result else 'None'}")
                                    
                                    if ocr_result and len(ocr_result) > 0:
                                        # New PaddleOCR format - extract from 'rec_texts' field
                                        page_result = ocr_result[0]
                                        if 'rec_texts' in page_result and page_result['rec_texts']:
                                            plate_text = ' '.join(page_result['rec_texts'])
                                            print(f"Found texts: {page_result['rec_texts']}")
                                        elif 'rec_scores' in page_result:
                                            print("No texts found in rec_texts, trying alternative parsing...")
                                            # Fallback to old format if needed
                                            pass
                                    
                                    plate_text = plate_text.strip()
                                    print(f"OCR result: '{plate_text}'")
                                    
                                    if plate_text:
                                        # Clean and format the detected text
                                        formatted_text = format_detected_plate(plate_text)
                                        
                                        # Parse the formatted plate text automatically
                                        plate_details = parsePlate(formatted_text)
                                        detected_plates.append({
                                            'text': plate_text,  # Original OCR text
                                            'formatted_text': formatted_text,  # Formatted text
                                            'confidence': float(confidence),
                                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                            'vehicle_details': plate_details
                                        })
                                except Exception as ocr_error:
                                    print(f"OCR error: {ocr_error}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                    except Exception as box_error:
                        print(f"Box processing error: {box_error}")
                        continue
        
        print(f"Final result: {len(detected_plates)} plates detected")
        return {"plates": detected_plates}
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Detection failed: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<filename>')
def static_files(filename):
    """Serve static files from templates folder"""
    return send_from_directory('templates', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        
        # Detect and read license plates
        result = detect_and_read_plate(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    return jsonify({"error": "Invalid file type. Please upload an image file."})

@app.route('/parse_plate', methods=['POST'])
def parse_plate():
    """Parse Sri Lankan license plate and return vehicle information"""
    try:
        data = request.get_json()
        if not data or 'plate' not in data:
            return jsonify({"error": "No plate data provided"}), 400
        
        plate_text = data['plate']
        result = parsePlate(plate_text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Parsing failed: {str(e)}"}), 500

@app.route('/export_to_csv', methods=['POST'])
def export_to_csv():
    """Export vehicle data to CSV file"""
    try:
        data = request.get_json()
        
        if not data or 'vehicle_data' not in data:
            return jsonify({"error": "No vehicle data provided"}), 400
        
        vehicle_data = data['vehicle_data']
        
        # Add image filename if provided
        if 'image_filename' in data:
            vehicle_data['image_filename'] = data['image_filename']
        
        # Append to CSV
        success = append_to_csv(vehicle_data)
        
        if success:
            return jsonify({
                "success": True, 
                "message": "Vehicle data successfully exported to CSV",
                "csv_file": app.config['CSV_FILE']
            })
        else:
            return jsonify({"error": "Failed to export data to CSV"}), 500
            
    except Exception as e:
        print(f"CSV export error: {str(e)}")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "models_loaded": True})

if __name__ == '__main__':
    print("Starting License Plate Detection Web App...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

