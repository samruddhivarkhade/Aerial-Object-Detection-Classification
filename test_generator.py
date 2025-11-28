from data_generator import DataGenerator

train_gen = DataGenerator("data/train")
val_gen = DataGenerator("data/validation")
test_gen = DataGenerator("data/test")

print("Train images:", len(train_gen))
print("Validation images:", len(val_gen))
print("Test images:", len(test_gen))

# Test loading one batch
for X_batch, y_batch in train_gen.load_batch(batch_size=32):
    print("Batch Loaded:", X_batch.shape, y_batch.shape)
    break
