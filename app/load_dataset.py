def get_dataset(file_dataset_path):
    dataset = []
    with open(file_dataset_path, 'r') as file:
        dataset = file.readlines()
        print(f"Loaded {len(dataset)} entries")
    return dataset