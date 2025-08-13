from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename
from healthy_recipes import healthy_recipes
import sys
from datetime import datetime
from PIL import Image
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# === Load Model ===
try:
    model = load_model('model_makanan_fleksibel_final.h5')
except Exception as e:
    sys.exit(f"Gagal memuat model: {e}")

# === Load Class Indices ===
try:
    with open('class_labels.pkl', 'rb') as f:
        class_indices = pickle.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
except Exception as e:
    sys.exit(f"Gagal memuat class_labels.pkl: {e}")

CONFIDENCE_THRESHOLD = 0.75

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base, ext = os.path.splitext(filename)
    return f"{base[:50]}_{timestamp}{ext}"

def preprocess_image(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Gagal memproses gambar: {str(e)}")

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/artikel')
def artikel():
    return render_template('artikel.html')

@app.route('/kalori')
def kalori():
    return render_template('kalori.html')

@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung'}), 400

    filepath = None
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)[0]

        top3_idx = predictions.argsort()[-3:][::-1]
        top3 = [
            {
                'label': idx_to_class.get(i, 'Tidak_Dikenali'),
                'confidence': float(predictions[i])
            }
            for i in top3_idx
        ]

        best = top3[0]
        best_label = best['label']
        best_conf = best['confidence']

        # Jika confidence terlalu rendah, return langsung dengan "tidak tersedia"
        if best_conf < CONFIDENCE_THRESHOLD or best_label == "Tidak_Dikenali":
            return jsonify({
                'class': 'Tidak_Dikenali',
                'confidence': float(best_conf),
                'healthy_recipe': "<p><i>Tidak tersedia</i></p>",
                'top3': top3
            })

        # Ambil resep sehat jika ada
        recipe_obj = healthy_recipes.get(best_label, {})
        if isinstance(recipe_obj, dict):
            bahan = recipe_obj.get('bahan', [])
            langkah = recipe_obj.get('langkah', [])
            sumber = recipe_obj.get('sumber', 'cookpad')

            if isinstance(bahan, list):
                bahan_html = "<ul style='margin-bottom: 1em;'>" + "".join(f"<li>{item}</li>" for item in bahan) + "</ul>"
            else:
                bahan_html = f"<p>{bahan}</p>"

            if isinstance(langkah, list):
                langkah_html = "<ol>" + "".join(f"<li>{step}</li>" for step in langkah) + "</ol>"
            else:
                langkah_html = f"<p>{langkah}</p>"

            sumber_html = f"<p style='margin-top: 1em;'><i>Sumber: <strong>{sumber.lower()}</strong></i></p>"

            recipe = f"""
            <div>
                <h4>Bahan:</h4>
                {bahan_html}
                <h4>Langkah:</h4>
                {langkah_html}
                {sumber_html}
            </div>
            """
        else:
            # Jika tidak ditemukan
            recipe = "<p><i>Tidak tersedia</i></p>"

        return jsonify({
            'class': best_label,
            'confidence': float(best_conf),
            'healthy_recipe': recipe,
            'top3': top3
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
