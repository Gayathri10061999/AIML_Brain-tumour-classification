from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import get_data_generators
from model_transfer import build_transfer_model
from config import MODEL_PATH, EPOCHS

train_dir = "../data/processed/train"
val_dir = "../data/processed/val"

train_gen, val_gen = get_data_generators(train_dir, val_dir)

model = build_transfer_model()

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)
