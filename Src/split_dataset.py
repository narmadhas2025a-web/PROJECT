import os
import random
import shutil


# PROJECT PATH


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Original dataset
SOURCE_DIR = os.path.join(PROJECT_DIR, "Dataset")

# Output dataset
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Dataset_split")


# SPLIT RATIO


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15



# RANDOM SEED


random.seed(42)



# CLASSES


classes = ["Non_PCOS", "PCOS"]



# CHECK DATASET


if not os.path.exists(SOURCE_DIR):
    print("ERROR: Dataset folder not found!")
    print(SOURCE_DIR)
    exit()



# PROCESS EACH CLASS


for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    # Check class folder
    if not os.path.exists(source_class_dir):
        print(f"ERROR: {class_name} folder not found!")
        print(source_class_dir)
        continue

    # Get image files
    images = [
        file
        for file in os.listdir(source_class_dir)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Shuffle images
    random.shuffle(images)

    # Total images
    total = len(images)

    # Calculate split sizes
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    # Split images
    train_images = images[:train_count]

    validation_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    
    # CREATE FOLDERS
  

    train_dir = os.path.join(
        OUTPUT_DIR,
        "train",
        class_name
    )

    validation_dir = os.path.join(
        OUTPUT_DIR,
        "validation",
        class_name
    )

    test_dir = os.path.join(
        OUTPUT_DIR,
        "test",
        class_name
    )

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

   
    # COPY TRAIN IMAGES
    

    for image_name in train_images:

        source = os.path.join(
            source_class_dir,
            image_name
        )

        destination = os.path.join(
            train_dir,
            image_name
        )

        shutil.copy2(source, destination)

    
    # COPY VALIDATION IMAGES
    
    for image_name in validation_images:

        source = os.path.join(
            source_class_dir,
            image_name
        )

        destination = os.path.join(
            validation_dir,
            image_name
        )

        shutil.copy2(source, destination)


    # COPY TEST IMAGES


    for image_name in test_images:

        source = os.path.join(
            source_class_dir,
            image_name
        )

        destination = os.path.join(
            test_dir,
            image_name
        )

        shutil.copy2(source, destination)

   
    # DISPLAY RESULT
    

    print()
    print("=" * 40)
    print(class_name)
    print("=" * 40)

    print(f"Total      : {total}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(validation_images)}")
    print(f"Test       : {len(test_images)}")



# COMPLETION MESSAGE


print()
print("=" * 50)
print("DATASET SPLITTING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print()
print("Output location:")
print(OUTPUT_DIR)