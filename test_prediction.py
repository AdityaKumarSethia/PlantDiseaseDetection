import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Paths relative to the project root
MODEL_PATH = "models/mobilenetv2_from_scratch.keras"
TEST_DIR = "test/test"
DATASET_PATH = "New Plant Diseases Dataset(Augmented)/valid" 
OUTPUT_FILE = "test_results_grid.png"

def main():
    # 1. Check Paths
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DIR):
        print(f"❌ Error: Test directory not found at {TEST_DIR}")
        return

    # 2. Get Class Names
    # We load the class names from the validation folder to ensure they match the model's training indices.
    print("--- Loading Class Labels ---")
    try:
        dummy_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_PATH,
            labels='inferred',
            label_mode='int',
            image_size=(224, 224),
            batch_size=32,
            shuffle=False,
            verbose=0
        )
        class_names = dummy_ds.class_names
        print(f"✅ Loaded {len(class_names)} classes.")
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return

    # 3. Load Model
    print(f"--- Loading Model from {MODEL_PATH} ---")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. Prepare Test Images
    # Get all .jpg/.jpeg/.png files
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(files)
    
    if num_images == 0:
        print("❌ No images found in test directory.")
        return

    print(f"--- Processing {num_images} images from {TEST_DIR} ---")

    # Calculate Grid Size (5 columns)
    cols = 5
    rows = math.ceil(num_images / cols)

    # Create Figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4.5))
    axes = axes.flatten()

    # 5. Loop, Predict, and Plot
    for i, filename in enumerate(files):
        img_path = os.path.join(TEST_DIR, filename)
        ax = axes[i]
        
        try:
            # Load & Preprocess
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            predicted_label = class_names[predicted_class_index]
            confidence = np.max(predictions[0]) * 100
            
            # Get "Actual" from filename
            actual_label = os.path.splitext(filename)[0]

            # Plot Image
            ax.imshow(img)
            
            # Set Title (Green if >80%, Red if <80%)
            title_color = 'green' if confidence > 80 else 'red'
            title_text = f"Act: {actual_label}\nPred: {predicted_label}\nConf: {confidence:.1f}%"
            
            ax.set_title(title_text, color=title_color, fontsize=9)
            ax.axis('off')
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 6. Clean up empty slots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    # 7. Save Result
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"\n✅ Success! Prediction grid saved as '{OUTPUT_FILE}'")
    print("You can now include this image in your project report.")

if __name__ == "__main__":
    main()