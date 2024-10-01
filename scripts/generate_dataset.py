from data.create_data import generate_dataset

def main():
    generate_dataset(num_images=10, output_dir="enhanced_synthetic_dataset/")

if __name__ == "__main__":
    main()