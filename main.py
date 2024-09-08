from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

def load_and_prepare_data(file_path='places.csv'):
    # Load the CSV file
    places = pd.read_csv(file_path)

    # Select necessary columns
    places = places[['Zone', 'State', 'City', 'Name', 'Type', 'Establishment Year',
                     'time needed to visit in hrs', 'Google review rating',
                     'Entrance Fee in INR', 'Airport with 50km Radius',
                     'Significance', 'DSLR Allowed',
                     'Number of google review in lakhs']]

    # Replace non-numeric establishment years with a fixed value
    def replace_invalid_year(value, replacement_year=1200):
        if isinstance(value, str):
            if any(term in value.lower() for term in ['century', 'ancient', 'thousand']):
                return replacement_year
            elif value.isdigit():
                return int(value)
            else:
                return replacement_year
        try:
            return int(value)
        except ValueError:
            return replacement_year

    places['Establishment Year'] = places['Establishment Year'].apply(replace_invalid_year)

    # Normalize the numeric columns
    numeric_cols = ['Establishment Year', 'time needed to visit in hrs',
                    'Google review rating', 'Entrance Fee in INR',
                    'Number of google review in lakhs']

    scaler = MinMaxScaler()
    places[numeric_cols] = scaler.fit_transform(places[numeric_cols])

    # Convert all columns to strings for concatenation
    for col in numeric_cols:
        places[col] = places[col].astype(str)

    # Create the 'tags' column
    places['tags'] = (places['Zone'].str.lower() + " " +
                      places['State'].str.lower() + " " +
                      places['City'].str.lower() + " " +
                      places['Type'].str.lower() + " " +
                      places['Airport with 50km Radius'].str.lower() + " " +
                      places['Significance'].str.lower() + " " +
                      places['DSLR Allowed'].str.lower() + " " +
                      places['Establishment Year'] + " " +
                      places['time needed to visit in hrs'] + " " +
                      places['Google review rating'] + " " +
                      places['Entrance Fee in INR'] + " " +
                      places['Number of google review in lakhs'])

    # Set 'Name' as the index
    places.set_index('Name', inplace=True)

    return places

def vectorize_and_cluster(df, n_clusters=20):
    # Vectorize the 'tags' column using TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    return tfidf_matrix, kmeans

def find_cluster_for_place(place_name, df):
    # Find the cluster of the input place
    if place_name not in df.index:
        return None  # Return None if place is not found

    place_cluster = df.loc[place_name, 'cluster']

    # Get all places in the same cluster
    places_in_cluster = df[df['cluster'] == place_cluster].index.tolist()

    return places_in_cluster

@app.route('/api/cluster', methods=['GET'])
def get_cluster():
    place_name = request.args.get('place_name')
    if not place_name:
        return jsonify({'error': 'Place name is required'}), 400

    try:
        # Load and prepare data
        df = load_and_prepare_data()

        # Vectorize and cluster the data
        tfidf_matrix, kmeans_model = vectorize_and_cluster(df)

        # Find the entire cluster for the place
        cluster_places = find_cluster_for_place(place_name, df)

        if cluster_places is None:
            return jsonify({'error': f"Place '{place_name}' not found."}), 404

        return jsonify({'place': place_name, 'cluster': cluster_places}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
