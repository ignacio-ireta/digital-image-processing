The code works as is. I need you to remove all the documentation, comments, and error handling to make it look more informal.


import os
import random
import time
import datetime
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, confusion_matrix, classification_report

# Define paths and constants
# Change this line:
BASE_PATH = os.path.join(os.getcwd(), 'Fruits')
# To this:
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fruits')
FRUIT_CLASSES = ['Aguacate', 'Berenjena', 'Mandarina', 'Manzana', 'Sandia']
TRAIN_SIZE = 42
TEST_SIZE = 18
BETA_VALUES = [0.5, 1.0, 2.0]
TARGET_SCORE = 0.95  # 95% threshold


def load_and_split_data(base_path, fruit_classes):
    """Load image paths and split into training and test sets."""
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    for class_idx, fruit_class in enumerate(fruit_classes):
        class_path = os.path.join(base_path, fruit_class)
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                      if f.endswith('.jpg')]
        
        # Validate image count
        if len(image_files) < TRAIN_SIZE + TEST_SIZE:
            print(f"Warning: {fruit_class} has only {len(image_files)} images, expected {TRAIN_SIZE + TEST_SIZE}")
        
        # Shuffle images for randomness
        random.shuffle(image_files)
        
        # Split into training and test sets
        train_paths.extend(image_files[:TRAIN_SIZE])
        train_labels.extend([class_idx] * min(TRAIN_SIZE, len(image_files[:TRAIN_SIZE])))
        
        test_paths.extend(image_files[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE])
        test_labels.extend([class_idx] * min(TEST_SIZE, len(image_files[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE])))
    
    return train_paths, train_labels, test_paths, test_labels


def extract_hu_moments(image_path):
    """Extract Hu moments from an image."""
    try:
        # Read image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return np.zeros(7)  # Return zeros if image can't be read
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        
        # Calculate Hu moments
        hu_moments = cv2.HuMoments(moments)
        
        # Log transform to improve numerical stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        return hu_moments.flatten()
    
    except Exception as e:
        print(f"Error extracting Hu moments from {image_path}: {e}")
        return np.zeros(7)  # Return zeros on error


def extract_features_from_paths(image_paths, feature_extractor=extract_hu_moments):
    """Extract features from a list of image paths."""
    features = []
    for path in image_paths:
        features.append(feature_extractor(path))
    return np.array(features)


def extract_enhanced_features(image_path):
    """Extract enhanced features (Hu moments + color histograms)."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return np.zeros(7 + 16*3)  # Return zeros if image can't be read
        
        # Extract Hu moments from grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        
        # Extract color histograms from HSV image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_bins = 16
        s_bins = 16
        v_bins = 16
        h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
        
        # Combine features
        combined_features = np.concatenate([hu_moments.flatten(), h_hist, s_hist, v_hist])
        
        return combined_features
    
    except Exception as e:
        print(f"Error extracting enhanced features from {image_path}: {e}")
        return np.zeros(7 + 16*3)  # Return zeros on error


def train_classifier(train_features, train_labels, classifier_type='svm'):
    """Train a classifier on the given features and labels."""
    if classifier_type == 'svm':
        classifier = SVC(kernel='rbf', probability=True)
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Train the classifier
    pipeline.fit(train_features, train_labels)
    
    return pipeline


def tune_classifier(train_features, train_labels, classifier_type='svm', beta=1.0):
    """Tune classifier hyperparameters using grid search."""
    if classifier_type == 'svm':
        classifier = SVC(probability=True)
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'classifier__kernel': ['rbf', 'poly', 'sigmoid']
        }
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier()
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]  # Manhattan or Euclidean distance
        }
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Create a pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Define scoring function based on beta
    def fbeta_weighted(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=beta, average='weighted')
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring=fbeta_weighted, verbose=1
    )
    grid_search.fit(train_features, train_labels)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_classifier(classifier, test_features, test_labels, beta_values):
    """Evaluate classifier using F-beta scores."""
    predictions = classifier.predict(test_features)
    
    # Calculate F-beta scores for different beta values
    scores = {}
    for beta in beta_values:
        score = fbeta_score(test_labels, predictions, beta=beta, average='weighted')
        scores[beta] = score
        print(f"F-{beta} Score: {score:.4f}")
    
    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=FRUIT_CLASSES))
    
    return scores


def main():
    """Main workflow for fruit classification."""
    start_time = time.time()
    print(f"Starting fruit classification at {datetime.datetime.now()}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load and split data
    print("\nLoading and splitting data...")
    train_paths, train_labels, test_paths, test_labels = load_and_split_data(BASE_PATH, FRUIT_CLASSES)
    print(f"Training set: {len(train_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    # Extract basic Hu moment features
    print("\nExtracting Hu moment features...")
    train_features = extract_features_from_paths(train_paths)
    test_features = extract_features_from_paths(test_paths)
    
    # Train classifier
    print("\nTraining classifier with Hu moments...")
    classifier = train_classifier(train_features, train_labels)
    
    # Evaluate classifier
    print("\nEvaluating classifier with Hu moments...")
    basic_scores = evaluate_classifier(classifier, test_features, test_labels, BETA_VALUES)
    
    # Check if any F-beta score exceeds the target
    if any(score >= TARGET_SCORE for score in basic_scores.values()):
        print(f"\nSuccess! At least one F-beta score exceeds {TARGET_SCORE*100}% with basic Hu moments.")
    else:
        print(f"\nBasic Hu moments did not achieve {TARGET_SCORE*100}% F-beta score. Trying enhanced features...")
        
        # Extract enhanced features
        print("\nExtracting enhanced features (Hu moments + color histograms)...")
        train_enhanced = extract_features_from_paths(train_paths, extract_enhanced_features)
        test_enhanced = extract_features_from_paths(test_paths, extract_enhanced_features)
        
        # Tune and train classifier with enhanced features
        print("\nTuning and training classifier with enhanced features...")
        tuned_classifier = tune_classifier(train_enhanced, train_labels, beta=1.0)
        
        # Evaluate tuned classifier
        print("\nEvaluating classifier with enhanced features...")
        enhanced_scores = evaluate_classifier(tuned_classifier, test_enhanced, test_labels, BETA_VALUES)
        
        # Check if any F-beta score exceeds the target
        if any(score >= TARGET_SCORE for score in enhanced_scores.values()):
            print(f"\nSuccess! At least one F-beta score exceeds {TARGET_SCORE*100}% with enhanced features.")
        else:
            print(f"\nWarning: Could not achieve {TARGET_SCORE*100}% F-beta score even with enhanced features.")
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()