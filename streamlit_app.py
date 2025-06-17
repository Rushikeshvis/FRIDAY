import streamlit as st
import torch
from PIL import Image
import pandas as pd

# Load YOLO model
model = torch.hub.load('ultralytics/yolo11', 'custom', path='Yolo11_Best.pt')

# Load coral data
coral_data = pd.read_csv('CatBoost_Data.csv')

# Define coral species
coral_species = [
    'Acanthastrea echinata', 'Acropora abrotanoides', 'Acropora cervicornis',
    'Acropora clathrata', 'Acropora humilis', 'Acropora hyacinthus',
    'Acropora palmata', 'Acropora prolifera', 'Agaricia fragilis',
    'Agaricia lamarcki', 'Astrea curta', 'Balanophyllia elegans',
    'Bernardpora stutchburyi', 'Blastomussa omanensis', 'Carijoa riisei',
    'Cladocora arbuscula', 'Cladocora caespitosa', 'Cladopsammia gracilis',
    'Coeloseris mayeri', 'Corallium rubrum', 'Coscinaraea monile',
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
st.title('Coral Classification with YOLO11')

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(" ")
    st.write("Classifying...")
    
    # Perform detection
    results = model(image)
    
    # Filter results for coral species
    coral_results = [res for res in results.xyxy[0] if res[-1] in coral_species]
    
    if coral_results:
        # Display the most confident classification
        best_result = max(coral_results, key=lambda x: x[4])
        st.write(f"Most confident classification: {coral_species[int(best_result[-1])]} with confidence {best_result[4]:.2f}")
        
        # Display segment of the image
        x1, y1, x2, y2 = map(int, best_result[:4])
        cropped_image = image.crop((x1, y1, x2, y2))
        st.image(cropped_image, caption='Segment of the Image', use_column_width=True)
    else:
        st.write("No coral species detected.")