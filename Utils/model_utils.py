import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from collections import Counter
import torch
import torch.nn as nn
import os
import torch.optim as optim

# Define the ImprovedTagClassifier class for tag prediction
class ImprovedTagClassifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.4):
        super(ImprovedTagClassifier, self).__init__()
        
        # First hidden layer: transforms input features to 512 dimensions
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Normalizes the output
        
        # Second hidden layer: reduces from 512 to 256 dimensions
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  # Normalizes again
        
        # Third hidden layer: further reduces to 128 dimensions
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Another normalization
        
        # Output layer: maps 128 dimensions to the number of classes
        self.fc4 = nn.Linear(128, output_size)
        
        # Tools to prevent overfitting and improve learning
        self.dropout = nn.Dropout(dropout_rate)  # Randomly drops some data
        self.leaky_relu = nn.LeakyReLU(0.1)  # Activation function with a small slope
        
        # Skip connection: connects layer 1 directly to layer 3
        self.skip1_3 = nn.Linear(512, 128)
        
        # Set up the initial weights for better training
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Loop through all parts of the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use a special method to set weights for linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    # Set biases to zero
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # Set batch norm weights to 1 and biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First block: process input through the first layer
        x1 = self.fc1(x)
        x1 = self.bn1(x1)  # Normalize
        x1 = self.leaky_relu(x1)  # Activate
        x1 = self.dropout(x1)  # Drop some data to prevent overfitting
        
        # Second block: process through the second layer
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)  # Normalize
        x2 = self.leaky_relu(x2)  # Activate
        x2 = self.dropout(x2)  # Drop some data
        
        # Third block: process with a skip connection
        x3 = self.fc3(x2)
        skip_x1 = self.skip1_3(x1)  # Skip connection from first layer
        x3 = x3 + skip_x1  # Add the skip connection
        x3 = self.bn3(x3)  # Normalize
        x3 = self.leaky_relu(x3)  # Activate
        x3 = self.dropout(x3)  # Drop some data
        
        # Final output: get the class predictions
        output = self.fc4(x3)
        return output

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # Weights for each class
        self.gamma = gamma    # Focus on hard examples
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        # Calculate basic cross-entropy loss
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # Adjust loss
        
        # Combine losses based on reduction type
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
class MultiLevelTagClassifier:
    def __init__(self, device='cuda'):
        # Use GPU
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}           # Store models for each parent tag
        self.preprocessors = {}    # Store preprocessing tools
        self.label_encoders = {}   # Store label encoders
        
        # Define tag groups
        self.tag_hierarchy = {
            'DIV': ['DIV', 'LIST', 'CARD', 'FORM'],
            'P': ['P', 'LABEL', 'LI', 'A'],
            'INPUT': ['INPUT', 'DROPDOWN'],
            'ICON': ['ICON', 'CHECKBOX', 'RADIO'],
        }
        print(f"Using device: {self.device}")
    
    def prepare_data_for_subtask(self, df, parent_tag, subtags):
        # Get only the data for this parent tag’s subtags
        filtered_df = df[df['tag'].isin(subtags)].copy()
        print(f"\n=== Preparing data for {parent_tag} sub-classification ===")
        print(f"Subtags: {subtags}")
        print(f"Total samples: {len(filtered_df)}")
        print(f"Distribution: \n{filtered_df['tag'].value_counts()}")
        
        if len(filtered_df) == 0:
            print(f"No data found for {parent_tag} subtags!")
            return None, None, None, None, None, None
        
        y = filtered_df["tag"]  # Target tags
        X = filtered_df.drop(columns=["tag"])  # Features
        
        # Define which columns are categories and numerical features
        categorical_cols = ['type', 'prev_sibling_html_tag', 'child_1_html_tag', 'child_2_html_tag', 'parent_tag_html']
        continuous_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Add missing columns with default values
        missing_cols = [col for col in categorical_cols + continuous_cols if col not in X.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in data for {parent_tag}")
            for col in missing_cols:
                X[col] = 'unknown' if col in categorical_cols else 0
        
        # Process categories
        X[categorical_cols] = X[categorical_cols].astype(str).fillna('unknown')
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = ohe.fit_transform(X[categorical_cols])
        
        # Process continous features
        imputer = SimpleImputer(strategy='median')
        X_continuous_imputed = imputer.fit_transform(X[continuous_cols])
        scaler = StandardScaler()
        X_continuous_scaled = scaler.fit_transform(X_continuous_imputed)
        X_processed = np.concatenate([X_cat_encoded, X_continuous_scaled], axis=1)
        
        # Encode target tags
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Boost rare classes by copying them
        class_counts = Counter(y_encoded)
        min_samples_threshold = max(10, len(subtags) * 3)
        rare_classes = [cls for cls, count in class_counts.items() if count < min_samples_threshold]
        
        for cls in rare_classes:
            idx = np.where(y_encoded == cls)[0]
            original_class_name = label_encoder.inverse_transform([cls])[0]
            samples_needed = min_samples_threshold - len(idx)
            print(f"Adding {samples_needed} copies to class '{original_class_name}'")
            for _ in range(samples_needed):
                sample_idx = np.random.choice(idx)
                new_sample = X_processed[sample_idx].copy()
                continuous_start = X_cat_encoded.shape[1]
                noise = np.random.normal(0, 0.05, size=X_continuous_scaled.shape[1])
                new_sample[continuous_start:] += noise
                X_processed = np.vstack([X_processed, new_sample])
                y_encoded = np.append(y_encoded, cls)
        
        # Bundle up preprocessing models
        preprocessors = {
            'ohe': ohe,
            'imputer': imputer,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'categorical_cols': categorical_cols,
            'continuous_cols': continuous_cols
        }
        return X_processed, y_encoded, preprocessors, categorical_cols, continuous_cols, label_encoder
    
    def train_subtask_model(self, X, y, preprocessors, parent_tag, epochs=100):
        # Split data into train, validation, and test sets
        print(f"\n=== Training {parent_tag} sub-classifier ===")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Validation set size: {X_val.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Balance classes
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        # Turn data into tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # Set up datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
        
        # Create and set up the model
        input_size = X_train.shape[1]
        output_size = len(np.unique(y))
        model = ImprovedTagClassifier(input_size, output_size).to(self.device)
        criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scaler = GradScaler()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 15
        counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                with autocast(device_type=self.device.type):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            model.eval()
            val_running_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    with autocast(device_type=self.device.type):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    val_running_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            val_loss = val_running_loss / len(val_loader)
            val_accuracy = accuracy_score(all_labels, all_preds)
            scheduler.step(val_loss)
            
            # Track progress
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model_state = model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        model.load_state_dict(best_model_state)
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch_y.cpu().numpy())
        
        test_accuracy = accuracy_score(test_labels, test_preds)
        print(f"\n{parent_tag} Model Test Accuracy: {test_accuracy:.4f}")
        print(f"\n{parent_tag} Classification Report:")
        print(classification_report(test_labels, test_preds, target_names=preprocessors['label_encoder'].classes_, zero_division=0))
        
        return model, (train_losses, val_losses, val_accuracies), test_accuracy
    
    def train_all_models(self, df_path, epochs=100):
        # Load and clean the main dataset
        print("Loading and cleaning data...")
        df = pd.read_csv(df_path)
        df.loc[(df["tag"] == "SPAN") & ((df["type"] == "RECTANGLE") | (df["type"] == "GROUP")), "tag"] = "DIV"
        children_cols = ['child_1_html_tag', 'child_2_html_tag']
        for col in children_cols:
            df[col] = df[col].apply(lambda x: "DIV" if isinstance(x, str) and '-' in x else x)
        for col in ['tag', 'prev_sibling_html_tag', 'child_1_html_tag', 'child_2_html_tag']:
            df[col] = df[col].str.upper()
        
        # Make a folder for models
        os.makedirs('../models/sub_classifiers', exist_ok=True)
        
        # Train a model for each parent tag
        for parent_tag, subtags in self.tag_hierarchy.items():
            print(f"\n{'='*60}")
            print(f"Training {parent_tag} sub-classifier")
            print(f"{'='*60}")
            result = self.prepare_data_for_subtask(df, parent_tag, subtags)
            if result[0] is None:
                print(f"Skipping {parent_tag} due to insufficient data")
                continue
            X, y, preprocessors, cat_cols, cont_cols, label_encoder = result
            model, training_history, test_accuracy = self.train_subtask_model(X, y, preprocessors, parent_tag, epochs)
            self.models[parent_tag] = model
            self.preprocessors[parent_tag] = preprocessors
            self.label_encoders[parent_tag] = label_encoder
            model_path = f'../models/sub_classifiers/{parent_tag.lower()}_classifier.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': X.shape[1],
                'output_size': len(np.unique(y)),
                'preprocessors': preprocessors,
                'test_accuracy': test_accuracy
            }, model_path)
            print(f"Saved {parent_tag} model to {model_path}")
            self.plot_training_history(training_history, parent_tag)
    
    def plot_training_history(self, history, parent_tag):
        # Plot training history (good function naming no need for commenting but here we go)
        train_losses, val_losses, val_accuracies = history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{parent_tag} Model: Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'{parent_tag} Model: Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../models/sub_classifiers/{parent_tag.lower()}_training_history.png')
        plt.close()
    
    def load_models(self, model_dir='../models/sub_classifiers'):
        # Load saved models
        for parent_tag in self.tag_hierarchy.keys():
            model_path = f'{model_dir}/{parent_tag.lower()}_classifier.pth'
            if os.path.exists(model_path):
                print(f"Loading {parent_tag} model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
                model = ImprovedTagClassifier(checkpoint['input_size'], checkpoint['output_size']).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models[parent_tag] = model
                self.preprocessors[parent_tag] = checkpoint['preprocessors']
                self.label_encoders[parent_tag] = checkpoint['preprocessors']['label_encoder']
                print(f"Loaded {parent_tag} model (Test Accuracy: {checkpoint['test_accuracy']:.4f})")
            else:
                print(f"Model file {model_path} not found!")
    
    def predict_hierarchical(self, sample_data, base_prediction):
        # Predict a tag using the right sub-classifier
        if base_prediction not in self.tag_hierarchy:
            return base_prediction, 1.0
        if base_prediction not in self.models:
            print(f"No sub-classifier found for {base_prediction}")
            return base_prediction, 1.0
        preprocessors = self.preprocessors[base_prediction]
        sample_df = pd.DataFrame([sample_data])
        cat_cols = preprocessors['categorical_cols']
        cont_cols = preprocessors['continuous_cols']
        
        # Add missing columns
        for col in cat_cols + cont_cols:
            if col not in sample_df.columns:
                sample_df[col] = 'unknown' if col in cat_cols else 0
        
        sample_df[cat_cols] = sample_df[cat_cols].astype(str).fillna('unknown')
        X_cat = preprocessors['ohe'].transform(sample_df[cat_cols])
        X_cont = preprocessors['imputer'].transform(sample_df[cont_cols])
        X_cont = preprocessors['scaler'].transform(X_cont)
        X_processed = np.concatenate([X_cat, X_cont], axis=1)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
        
        model = self.models[base_prediction]
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        predicted_label = preprocessors['label_encoder'].inverse_transform([predicted.cpu().numpy()[0]])[0]
        confidence = probabilities.max().item()
        return predicted_label, confidence