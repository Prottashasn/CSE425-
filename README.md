# Music Genre Classification using Variational Autoencoders (VAE)

A deep learning project for music genre classification and clustering using Variational Autoencoders (VAE) with multi-modal features (audio + lyrics). The project implements various clustering techniques for genre discovery.

## 🎵 Project Overview

This project implements a comprehensive pipeline for music genre classification using:
- **Variational Autoencoders (VAE)** for feature extraction and latent space learning
- **Multi-modal features**: Audio features (mel-spectrograms) + Lyrics embeddings
- **Multiple clustering algorithms**: K-Means, Agglomerative Clustering, DBSCAN
- **Comprehensive evaluation metrics**: Silhouette Score, Davies-Bouldin Index, Adjusted Rand Index

## ✨ Features

- 🎨 **Hybrid Feature Extraction**: Combines audio features with lyrics embeddings
- 📊 **Comprehensive EDA**: Exploratory data analysis with visualizations
- 🔬 **Multiple VAE Architectures**: Standard VAE and Convolutional VAE
- 📈 **Visualization Tools**: t-SNE plots, latent space visualizations, clustering results
- 🎯 **Conditional VAE**: Genre-conditioned latent representations for better disentanglement

## 📁 Project Structure

```
425 Project/
├── Datasets/                      # Dataset directory (audio files and metadata)
├── data_preprocessing.ipynb      # Data preprocessing pipeline
├── EDA.ipynb                     # Exploratory Data Analysis
├── vae_clustering-easy.ipynb     # Basic VAE clustering
├── enhanced_vae_clustering-mdim.ipynb  # Enhanced VAE with CNN
├── -hard.py                      # Conditional VAE implementation
├── run_data_preprocessing.py     # Script to run preprocessing
├── results/                      # Generated visualizations and plots
│   ├── language_distribution.png
│   ├── genre_distribution.png
│   ├── latent_space_genre.png
│   ├── latent_space_language.png
│   ├── training_loss.png
│   └── ...
├── mel_spectrograms.npy          # Preprocessed mel-spectrograms
├── audio_features.npy            # Extracted audio features
├── lyrics_embeddings.npy         # Lyrics TF-IDF embeddings
├── hybrid_features.npy           # Combined audio + lyrics features
├── labels.npy                    # Genre labels
├── genre_mapping.pkl             # Genre to label mapping
└── preprocessing_info.pkl        # Preprocessing metadata
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see below)

### Installation

1. **Clone the repository** (or download the project):
```bash
git clone <repository-url>
cd "425 Project"
```

2. **Install required packages**:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch torchvision jupyter
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Workflow

The project follows this workflow:

1. **Data Preprocessing** → 2. **EDA** → 3. **VAE Training** → 4. **Clustering** → 5. **Evaluation**

### Step 1: Data Preprocessing

Run the preprocessing notebook to extract features from audio files:

```bash
jupyter notebook data_preprocessing.ipynb
```

Or use the Python script:
```bash
python run_data_preprocessing.py
```

This will generate:
- `mel_spectrograms.npy`: Mel-spectrograms for CNN-based VAE
- `audio_features.npy`: Traditional audio features
- `lyrics_embeddings.npy`: TF-IDF based lyrics embeddings
- `hybrid_features.npy`: Combined audio + lyrics features
- `labels.npy`: Genre labels
- `genre_mapping.pkl`: Genre mappings

### Step 2: Exploratory Data Analysis

Explore your dataset with the EDA notebook:

```bash
jupyter notebook EDA.ipynb
```

This notebook provides:
- Dataset overview and statistics
- Language and genre distributions
- Lyrics analysis
- Audio file properties analysis
- Preprocessed features analysis
- Data quality checks

All visualizations are saved to the `results/` directory.

### Step 3: VAE Clustering

Choose one of the VAE implementations:

#### Option A: Basic VAE Clustering
```bash
jupyter notebook vae_clustering-easy.ipynb
```

#### Option B: Enhanced VAE with Convolutional Architecture
```bash
jupyter notebook enhanced_vae_clustering-mdim.ipynb
```

#### Option C: Conditional VAE (Python Script)
```bash
python -hard.py
```

### Step 4: View Results

All generated plots and visualizations are saved in the `results/` directory:
- Clustering visualizations (t-SNE plots)
- Latent space visualizations by genre and language
- Training loss curves
- Reconstruction examples
- Cluster distribution plots

## 🔧 Configuration

### Audio Processing Parameters

In `data_preprocessing.ipynb`, you can adjust:
- `TARGET_SR = 22050`: Target sample rate
- `DURATION = 3.0`: Duration in seconds (for 3-second clips)
- `N_MELS = 128`: Number of mel filterbanks
- `N_FFT = 2048`: FFT window size
- `HOP_LENGTH = 512`: Hop length for STFT

### VAE Parameters

In the VAE notebooks, you can modify:
- `latent_dim`: Dimension of latent space (default: 32)
- `hidden_dims`: Hidden layer dimensions
- `beta`: Beta-VAE parameter for disentanglement
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs

## 🛠️ Technologies Used

- **Python**: Core programming language
- **PyTorch**: Deep learning framework for VAE implementation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities (clustering, metrics, preprocessing)
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Signal processing for audio analysis
- **Jupyter Notebook**: Interactive development environment

## 📈 Results

The project generates various visualizations and metrics:

- **Clustering Performance**: Silhouette scores, Davies-Bouldin index, Adjusted Rand Index
- **Latent Space Visualizations**: t-SNE plots showing genre and language separability
- **Training Metrics**: Loss curves for VAE training
- **Reconstruction Quality**: Examples of reconstructed mel-spectrograms
- **Distribution Analysis**: Genre and language distributions

All results are saved in the `results/` directory.

## 🎯 Key Features of Implementation

### 1. Multi-Modal Feature Extraction
- **Audio Features**: Mel-spectrograms, spectral features, zero-crossing rate, energy
- **Lyrics Features**: TF-IDF embeddings with unigrams and bigrams
- **Hybrid Features**: Concatenated and standardized audio + lyrics features

### 2. VAE Architectures
- **Standard VAE**: Fully connected layers for feature-based input
- **Convolutional VAE**: CNN-based architecture for mel-spectrogram processing
- **Conditional VAE**: Genre-conditioned latent representations

### 3. Clustering Methods
- **K-Means**: Partition-based clustering
- **Agglomerative Clustering**: Hierarchical clustering
- **DBSCAN**: Density-based clustering

### 4. Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Davies-Bouldin Index**: Lower is better (cluster quality)
- **Adjusted Rand Index**: Compares clustering to ground truth labels

## 🔍 File Descriptions

- **`data_preprocessing.ipynb`**: Complete preprocessing pipeline for audio and lyrics
- **`EDA.ipynb`**: Comprehensive exploratory data analysis
- **`vae_clustering-easy.ipynb`**: Basic VAE implementation for clustering
- **`enhanced_vae_clustering-mdim.ipynb`**: Enhanced VAE with CNN architecture
- **`-hard.py`**: Conditional VAE implementation with PyTorch
- **`run_data_preprocessing.py`**: Script version of preprocessing notebook

## 📝 Notes

- The preprocessing step is **required** before running any VAE notebooks
- Audio files are processed as 3-second clips at 22.05 kHz sample rate
- All preprocessed features are saved as `.npy` files for efficient loading

## 🚧 Future Improvements

- [ ] Add support for more audio formats
- [ ] Implement transformer-based lyrics embeddings
- [ ] Add real-time audio classification
- [ ] Implement active learning for dataset expansion
- [ ] Add web interface for interactive exploration
- [ ] Support for more languages and genres
- [ ] Hyperparameter optimization pipeline

## 📄 License

[Specify your license here]

## 👥 Authors

[Add your name/team information here]

## 🙏 Acknowledgments

- Libraries and frameworks used
- Any other acknowledgments

## 📧 Contact

[Add contact information if needed]

---

**Note**: This project is part of a 425 course project. For questions or issues, please refer to the project documentation or contact the authors.

