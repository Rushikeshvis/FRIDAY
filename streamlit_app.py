import streamlit as st
import torch
from PIL import Image
import pandas as pd
from ultralytics import YOLO # Import YOLO from ultralytics
import os # Import os module to check for file existence

# --- Configuration for Logo ---
# IMPORTANT: Replace 'my_logo.png' with the actual filename of your logo image.
# Ensure your logo image file is in the same directory as this script.
LOGO_PATH = 'Friday.png'
LOGO_WIDTH = 400 # Adjust the width of the logo in pixels as needed
# --- End Configuration ---

# Load YOLO model
# Assuming Yolo11_Best.pt is a model trained with ultralytics framework
# and is in the same directory as streamlit_app.py
try:
    model = YOLO('Yolo11_Best.pt')
except Exception as e:
    st.error(f"Error loading YOLO model: {e}. Make sure 'Yolo11_Best.pt' is in the same directory and is a valid YOLO model file.")
    st.stop() # Stop the app if model loading fails

# Load coral data
try:
    coral_data = pd.read_csv('CatBoost_Data.csv')
except FileNotFoundError:
    st.error("Error: 'CatBoost_Data.csv' not found. Make sure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading 'CatBoost_Data.csv': {e}")
    st.stop()


# Define coral species (ensure this matches your model's classes)
# IMPORTANT: The class names in this list MUST correspond to the integer IDs
# that your YOLO11 model outputs. If your model outputs class ID '0' and that
# corresponds to 'Acanthastrea echinata', then this mapping is correct.
# If your model outputs a different order, you will need to adjust this list.
coral_species = [
    'Acanthastrea echinata', 'Acropora abrotanoides', 'Acropora cervicornis',
    'Acropora clathrata', 'Acropora humilis', 'Acropora hyacinthus',
    'Acropora palmata', 'Acropora prolifera', 'Agaricia fragilis',
    'Agaricia lamarcki', 'Astrea curta', 'Balanophyllia elegans',
    'Bernardpora stutchburyi', 'Blastomussa omanensis', 'Carijoa riisei',
    'Cladocora arbuscula', 'Cladocora caespitosa', 'Cladocora caespitosa',
    'Cladopsammia gracilis', 'Coeloseris mayeri', 'Corallium rubrum', 'Coscinaraea monile',
    'Culicia tenella', 'Cycloseris mokai', 'Cycloseris vaughani',
    'Dendrogyra cylindrus', 'Dendrophyllia ramea', 'Dichocoenia stokesi',
    'Dichocoenia stokesii', 'Diploastrea heliopora', 'Dipsastraea favus',
    'Dipsastraea pallida', 'Duncanopsammia peltata', 'Echinopora hirsutissima',
    'Echinopora lamellosa', 'Euphyllia cristata', 'Euphyllia glabrescens',
    'Eusmilia fastigiata', 'Favia fragum', 'Fungia fungites',
    'Gardineroseris planulata', 'Goniastrea stelligera', 'Goniopora lobata',
    'Heliopora coerulea', 'Helioseris cucullata', 'Herpolitha limax',
    'Homophyllia australis', 'Hydnophora exesa', 'Isophyllia rigida',
    'Isophyllia sinuosa', 'Isopora palifera', 'Leptastrea purpurea',
    'Leptoria phrygia', 'Leptoseris mycetoseroides', 'Madracis auretenra',
    'Madracis decactis', 'Meandrina meandrites', 'Merulina scabricula',
    'Montastraea cavernosa', 'Montipora capitata', 'Montipora flabellata',
    'Oculina patagonica', 'Orbicella annularis', 'Orbicella franksi',
    'Oulophyllia crispa', 'Pachyseris rugosa', 'Pachyseris speciosa',
    'Paracyathus stearnsii', 'Paragoniastrea russelli', 'Paramontastraea peresi',
    'Pavona clavus', 'Pavona duerdeni', 'Pavona maldivensis',
    'Pavona varians', 'Pavona venosa', 'Platygyra daedalea',
    'Plerogyra sinuosa', 'Plesiastrea versipora', 'Pocillopora aliciae',
    'Pocillopora damicornis', 'Pocillopora grandis', 'Pocillopora verrucosa',
    'Porites astreoides', 'Porites compressa', 'Porites cylindrica',
    'Porites lobata', 'Psammocora contigua', 'Pseudodiploria clivosa',
    'Sandalolitha robusta', 'Scolymia cubensis', 'Seriatopora hystrix',
    'Siderastrea radians', 'Siderastrea siderea', 'Sinularia brassica',
    'Stephanocoenia intersepta', 'Stylocoeniella armata', 'Stylophora pistillata',
    'Trachyphyllia geoffroyi', 'Tubastraea aurea', 'Tubastraea diaphana',
    'Tubastraea micranthus', 'Turbinaria reniformis'
]


# Streamlit app
# Display the logo at the top of the app
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=LOGO_WIDTH)
else:
    st.warning(f"Logo image not found at '{LOGO_PATH}'. Please ensure it's in the same directory as streamlit_app.py.")

st.title('Fringing Reef Identification and Detection Automation using Yolo11')
st.markdown("Upload an image of coral for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(" ")
    st.write("Classifying...")

    # Perform detection
    results = model(image) # This runs inference

    detected_objects = []
    # Loop through detections for the first (and likely only) image
    if results and len(results) > 0:
        for r in results: # 'r' is a Results object for one image
            if r.boxes and r.boxes.xyxy is not None:
                for box_data in r.boxes:
                    # box_data has .xyxy, .conf, .cls
                    x1, y1, x2, y2 = box_data.xyxy[0].tolist() # Convert tensor to list
                    conf = box_data.conf[0].item() # Convert tensor to float
                    cls_id = int(box_data.cls[0].item()) # Convert tensor to int

                    detected_objects.append({
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': coral_species[cls_id] if cls_id < len(coral_species) else "Unknown"
                    })

    # Filter results for coral species
    # Here, we assume ALL detected classes are coral species if their ID is in range
    # You might want a specific mapping if your YOLO model detects non-coral objects too
    coral_results = [obj for obj in detected_objects if obj['class_name'] in coral_species]


    if coral_results:
        # Display the most confident classification
        best_result = max(coral_results, key=lambda x: x['confidence'])
        st.success(f"Most confident classification: **{best_result['class_name']}** with confidence **{best_result['confidence']:.2f}**")

        # Display segment of the image
        x1, y1, x2, y2 = best_result['x1'], best_result['y1'], best_result['x2'], best_result['y2']
        cropped_image = image.crop((x1, y1, x2, y2))
        st.image(cropped_image, caption=f"Detected: {best_result['class_name']}", use_column_width=True)

        st.subheader("All Detections:")
        # Sort by confidence in descending order for display
        for i, res in enumerate(sorted(coral_results, key=lambda x: x['confidence'], reverse=True)):
            st.write(f"- **{res['class_name']}** (Confidence: {res['confidence']:.2f})")
            if i < 5: # Show top 5 cropped images for clarity, adjust as needed
                img_to_show = image.crop((res['x1'], res['y1'], res['x2'], res['y2']))
                st.image(img_to_show, width=200) # Smaller image for multiple detections

    else:
        st.info("No coral species detected in the image based on the model's predictions.")
