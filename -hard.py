"""
Conditional VAE for Multi-Modal Clustering
Combines audio, lyrics, and genre information for disentangled representation learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ConditionalVAE(nn.Module):
    """
    Conditional VAE with disentangled latent representations
    Conditions on genre information to learn better representations
    """
    
    def __init__(self, input_dim: int, condition_dim: int, latent_dim: int = 32, 
                 hidden_dims: List[int] = [256, 128, 64], beta: float = 1.0):
        """
        Args:
            input_dim: Dimension of input features (audio + lyrics)
            condition_dim: Dimension of condition (genre encoding)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            beta: Beta-VAE parameter for disentanglement
        """
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: input + condition -> latent
        encoder_layers = []
        prev_dim = input_dim + condition_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder: latent + condition -> reconstruction
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming normalized inputs
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input with condition to latent distribution"""
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent with condition to reconstruction"""
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar


class MultiModalDataset(Dataset):
    """Dataset for multi-modal data (audio, lyrics, genre)"""
    
    def __init__(self, audio_features: np.ndarray, lyrics_features: np.ndarray, 
                 genre_labels: np.ndarray):
        """
        Args:
            audio_features: Audio feature vectors (n_samples, audio_dim)
            lyrics_features: Lyrics feature vectors (n_samples, lyrics_dim)
            genre_labels: Genre labels (n_samples,)
        """
        self.audio_features = torch.FloatTensor(audio_features)
        self.lyrics_features = torch.FloatTensor(lyrics_features)
        self.genre_labels = genre_labels
        
        # Combine audio and lyrics
        self.combined_features = torch.cat([self.audio_features, self.lyrics_features], dim=1)
        
        # Encode genre as one-hot
        self.genre_encoder = LabelEncoder()
        self.genre_encoded = self.genre_encoder.fit_transform(genre_labels)
        self.genre_onehot = torch.FloatTensor(
            np.eye(len(self.genre_encoder.classes_))[self.genre_encoded]
        )
        
    def __len__(self):
        return len(self.combined_features)
    
    def __getitem__(self, idx):
        return {
            'features': self.combined_features[idx],
            'audio': self.audio_features[idx],
            'lyrics': self.lyrics_features[idx],
            'genre': self.genre_onehot[idx],
            'genre_label': self.genre_labels[idx]
        }


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
             logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """VAE loss: reconstruction + KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


def train_cvae(model: ConditionalVAE, dataloader: DataLoader, epochs: int = 100, 
               lr: float = 0.001, device: str = 'cpu') -> List[float]:
    """Train Conditional VAE"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            features = batch['features'].to(device)
            condition = batch['genre'].to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(features, condition)
            loss = vae_loss(recon, features, mu, logvar, beta=model.beta)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses


def extract_latent_representations(model: ConditionalVAE, dataloader: DataLoader, 
                                   device: str = 'cpu') -> np.ndarray:
    """Extract latent representations from trained VAE"""
    model.eval()
    latents = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            condition = batch['genre'].to(device)
            mu, _ = model.encode(features, condition)
            latents.append(mu.cpu().numpy())
    
    return np.vstack(latents)


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate cluster purity"""
    n_samples = len(y_true)
    unique_labels = np.unique(y_pred)
    
    purity = 0
    for label in unique_labels:
        cluster_mask = y_pred == label
        cluster_labels = y_true[cluster_mask]
        if len(cluster_labels) > 0:
            most_common = np.bincount(cluster_labels).max()
            purity += most_common
    
    return purity / n_samples


def evaluate_clustering(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate clustering performance"""
    metrics = {
        'Silhouette Score': silhouette_score(X, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'ARI': adjusted_rand_score(y_true, y_pred),
        'Cluster Purity': cluster_purity(y_true, y_pred)
    }
    return metrics


def plot_latent_space(latents: np.ndarray, labels: np.ndarray, title: str = "Latent Space", 
                     label_name: str = "Genre", save_path: Optional[str] = None):
    """Visualize latent space using t-SNE"""
    # Use t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, 
                         cmap='tab20', alpha=0.6, s=50)
    plt.colorbar(scatter, label=label_name)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    plt.close()


def plot_cluster_distribution(cluster_labels: np.ndarray, category_labels: np.ndarray, 
                              category_name: str = "Genre", save_path: Optional[str] = None):
    """Plot cluster distribution over categories"""
    unique_clusters = np.unique(cluster_labels)
    unique_categories = np.unique(category_labels)
    
    # Create contingency matrix
    contingency = np.zeros((len(unique_clusters), len(unique_categories)))
    for i, cluster in enumerate(unique_clusters):
        for j, category in enumerate(unique_categories):
            contingency[i, j] = np.sum((cluster_labels == cluster) & (category_labels == category))
    
    # Normalize by cluster size
    cluster_sizes = contingency.sum(axis=1, keepdims=True)
    contingency_norm = contingency / (cluster_sizes + 1e-8)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(contingency_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=unique_categories, yticklabels=[f'Cluster {c}' for c in unique_clusters],
                cbar_kws={'label': 'Proportion'})
    plt.title(f'Cluster Distribution over {category_name}', fontsize=14, fontweight='bold')
    plt.xlabel(category_name)
    plt.ylabel('Cluster')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    plt.close()


def plot_reconstruction_examples(model: ConditionalVAE, dataloader: DataLoader, 
                                 n_examples: int = 8, device: str = 'cpu', 
                                 save_path: Optional[str] = None):
    """Visualize reconstruction examples"""
    model.eval()
    examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            condition = batch['genre'].to(device)
            recon, _, _ = model(features, condition)
            
            for i in range(min(n_examples, len(features))):
                examples.append({
                    'original': features[i].cpu().numpy(),
                    'reconstructed': recon[i].cpu().numpy(),
                    'genre': batch['genre_label'][i]
                })
                if len(examples) >= n_examples:
                    break
            if len(examples) >= n_examples:
                break
    
    n_examples = len(examples)
    fig, axes = plt.subplots(2, n_examples, figsize=(2*n_examples, 4))
    if n_examples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, ex in enumerate(examples):
        # Original
        axes[0, i].plot(ex['original'][:100])  # Plot first 100 dimensions
        axes[0, i].set_title(f'Original\n{ex["genre"]}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Reconstructed
        axes[1, i].plot(ex['reconstructed'][:100])
        axes[1, i].set_title('Reconstructed')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    
    plt.suptitle('VAE Reconstruction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    plt.close()


def compare_clustering_methods(X: np.ndarray, y_true: np.ndarray, n_clusters: int) -> Dict[str, Dict[str, float]]:
    """Compare different clustering methods"""
    results = {}
    
    # 1. PCA + K-Means
    print("Running PCA + K-Means...")
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred_pca = kmeans_pca.fit_predict(X_pca)
    results['PCA + K-Means'] = evaluate_clustering(X_pca, y_true, y_pred_pca)
    
    # 2. Autoencoder + K-Means
    print("Running Autoencoder + K-Means...")
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = SimpleAutoencoder(X.shape[1], latent_dim=32).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=0.001)
    
    X_tensor = torch.FloatTensor(X).to(device)
    for epoch in range(50):
        optimizer.zero_grad()
        recon = ae(X_tensor)
        loss = F.mse_loss(recon, X_tensor)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        X_ae = ae.encoder(X_tensor).cpu().numpy()
    
    kmeans_ae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred_ae = kmeans_ae.fit_predict(X_ae)
    results['Autoencoder + K-Means'] = evaluate_clustering(X_ae, y_true, y_pred_ae)
    
    # 3. Direct spectral feature clustering (using raw features)
    print("Running Direct K-Means on raw features...")
    kmeans_direct = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred_direct = kmeans_direct.fit_predict(X)
    results['Direct K-Means'] = evaluate_clustering(X, y_true, y_pred_direct)
    
    return results


def main():
    """Main execution function"""
    print("=" * 60)
    print("Conditional VAE Multi-Modal Clustering")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # ========== DATA PREPARATION ==========
    print("\n[1/6] Preparing data...")
    
    # Create results folder
    os.makedirs('results', exist_ok=True)
    
    # Load preprocessed data
    print("Loading preprocessed data files...")
    audio_features = np.load('audio_features.npy')
    lyrics_embeddings = np.load('lyrics_embeddings.npy')
    labels = np.load('labels.npy')
    
    # Load genre mapping
    import pickle
    with open('genre_mapping.pkl', 'rb') as f:
        mappings = pickle.load(f)
        label_to_genre = mappings['label_to_genre']
        genre_to_label = mappings['genre_to_label']
    
    # Convert numeric labels to genre names
    genre_labels = np.array([label_to_genre[label] for label in labels])
    genres = sorted(genre_to_label.keys())
    
    # Create language labels based on dataset (Bangla vs English)
    # We'll infer from genre names - Bangla genres are capitalized, English are lowercase
    language_labels = np.array(['Bangla' if genre[0].isupper() else 'English' for genre in genre_labels])
    
    print(f"[OK] Dataset loaded: {len(audio_features)} samples")
    print(f"   Audio features shape: {audio_features.shape}")
    print(f"   Lyrics embeddings shape: {lyrics_embeddings.shape}")
    print(f"   Number of genres: {len(genres)}")
    print(f"   Genres: {', '.join(genres)}")
    
    # Create dataset (language labels kept separate for visualization only)
    dataset = MultiModalDataset(audio_features, lyrics_embeddings, genre_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Combined feature dimension: {dataset.combined_features.shape[1]}")
    print(f"Number of genres: {len(dataset.genre_encoder.classes_)}")
    
    # ========== TRAIN CONDITIONAL VAE ==========
    print("\n[2/6] Training Conditional VAE...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    input_dim = dataset.combined_features.shape[1]
    condition_dim = dataset.genre_onehot.shape[1]
    latent_dim = 32
    
    cvae = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        hidden_dims=[256, 128, 64],
        beta=1.0
    )
    
    train_losses = train_cvae(cvae, dataloader, epochs=100, lr=0.001, device=device)
    
    # ========== EXTRACT LATENT REPRESENTATIONS ==========
    print("\n[3/6] Extracting latent representations...")
    latents = extract_latent_representations(cvae, dataloader, device=device)
    print(f"Latent representations shape: {latents.shape}")
    
    # ========== CLUSTERING ==========
    print("\n[4/6] Performing clustering...")
    n_clusters = len(genres)
    
    # K-Means on VAE latents
    kmeans_vae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_vae = kmeans_vae.fit_predict(latents)
    
    # Encode genre labels for evaluation
    genre_encoded = dataset.genre_encoder.transform(genre_labels)
    
    # Evaluate VAE clustering
    metrics_vae = evaluate_clustering(latents, genre_encoded, cluster_labels_vae)
    print("\nVAE-based Clustering Metrics:")
    for metric, value in metrics_vae.items():
        print(f"  {metric}: {value:.4f}")
    
    # ========== COMPARE WITH BASELINES ==========
    print("\n[5/6] Comparing with baseline methods...")
    X_combined = dataset.combined_features.numpy()
    comparison_results = compare_clustering_methods(X_combined, genre_encoded, n_clusters)
    
    # Add VAE results to comparison
    comparison_results['Conditional VAE'] = metrics_vae
    
    # Print comparison
    print("\n" + "=" * 60)
    print("CLUSTERING COMPARISON RESULTS")
    print("=" * 60)
    methods = list(comparison_results.keys())
    metric_names = list(comparison_results[methods[0]].keys())
    
    # Create comparison table
    print(f"\n{'Method':<25} " + " ".join([f"{m:<15}" for m in metric_names]))
    print("-" * 80)
    for method in methods:
        print(f"{method:<25} ", end="")
        for metric in metric_names:
            print(f"{comparison_results[method][metric]:>14.4f} ", end="")
        print()
    
    # ========== VISUALIZATIONS ==========
    print("\n[6/6] Generating visualizations...")
    
    # 1. Latent space visualization
    print("  - Latent space plot...")
    plot_latent_space(latents, genre_encoded, 
                     title="Conditional VAE Latent Space (colored by Genre)",
                     label_name="Genre",
                     save_path="results/latent_space_genre.png")
    
    # Language visualization (language labels kept separate for visualization only)
    le_lang = LabelEncoder()
    lang_encoded = le_lang.fit_transform(language_labels)
    plot_latent_space(latents, lang_encoded,
                     title="Conditional VAE Latent Space (colored by Language)",
                     label_name="Language",
                     save_path="results/latent_space_language.png")
    
    # 2. Cluster distribution over genres
    print("  - Cluster distribution over genres...")
    plot_cluster_distribution(cluster_labels_vae, genre_labels,
                             category_name="Genre",
                             save_path="results/cluster_distribution_genre.png")
    
    # 3. Cluster distribution over languages
    print("  - Cluster distribution over languages...")
    plot_cluster_distribution(cluster_labels_vae, language_labels,
                             category_name="Language",
                             save_path="results/cluster_distribution_language.png")
    
    # 4. Reconstruction examples
    print("  - Reconstruction examples...")
    plot_reconstruction_examples(cvae, dataloader, n_examples=8, device=device,
                                save_path="results/reconstruction_examples.png")
    
    # 5. Training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Conditional VAE Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Analysis complete! All visualizations saved.")
    print("=" * 60)
    
    return {
        'model': cvae,
        'latents': latents,
        'cluster_labels': cluster_labels_vae,
        'metrics': metrics_vae,
        'comparison': comparison_results
    }


if __name__ == "__main__":
    results = main()

