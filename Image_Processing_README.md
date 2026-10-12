# Pet Segmentation and Breed Classification

A multi-task computer vision project that compares **Base U-Net** and **Attention U-Net** on the Oxford-IIIT Pet dataset. Each model shares an encoder between two tasks: separating the pet from the background and predicting its breed.

Developed as an academic project for **CSE428**.

## What the project includes

- Binary pet segmentation with foreground and boundary pixels merged into one foreground class.
- Classification across 37 pet breeds.
- Dataset exploration, mask overlays, and class-distribution plots.
- Training and validation curves for both model variants.
- Segmentation and classification evaluation.
- A separate notebook for inference using saved weights, without retraining.

## Technologies

Python, PyTorch, NumPy, pandas, Matplotlib, scikit-image, scikit-learn, and Google Colab.

## Included files

| File | Purpose |
| --- | --- |
| `CSE428_Final.ipynb` | Data preparation, model definitions, training, evaluation, and comparison |
| `CSE428_Evaluation.ipynb` | Load saved models and predict masks and breeds for uploaded images |
| `base_unet_weights (2).pth` | Supplied Base U-Net checkpoint |
| `attention_unet_weights (1).pth` | Supplied Attention U-Net checkpoint |

These are the original supplied filenames. For a cleaner repository, the checkpoints can be renamed to `base_unet_weights.pth` and `attention_unet_weights.pth`. The evaluation notebook supports these names and also detects similarly named checkpoint files.

## Dataset and preprocessing

The training notebook uses Oxford-IIIT Pet images, breed annotations, and trimap masks. Dataset files are **not included** in this project archive.

The notebook expects:

- `images/`: pet photographs.
- `annotations/trimaps/`: segmentation masks.
- `annotations/trainval.txt`: training and validation source list.
- `annotations/test.txt`: official test list.

RGB images are resized to **128 × 128**. Trimaps are converted into binary masks: foreground and boundary become `1`, while background becomes `0`. The conversion handles both label arrays and palette-decoded masks.

The official test partition is retained. The training/validation source partition is split into 80% training and 20% validation, stratified by breed, with `random_state=125`. Images are normalized using per-channel statistics from the training subset.

## Models and training configuration

| Setting | Implementation |
| --- | --- |
| Models | Base U-Net and Attention U-Net with a breed-classification head |
| Encoder channels | 16 → 32 → 64 → 128, followed by a 256-channel bottleneck |
| Classifier input | Final encoder feature map |
| Input resolution | 128 × 128 RGB |
| Batch size | 8 |
| Epochs | 25 per model in the training calls |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Segmentation losses | Binary cross-entropy with logits and Dice loss |
| Classification loss | Cross-entropy |
| Device | CUDA when available; otherwise CPU |

## Run inference without training

The supplied evaluation notebook is designed for **Google Colab**.

1. Open `CSE428_Evaluation.ipynb` in Colab.
2. Install any missing packages in a new cell:

   ```python
   %pip install torch numpy pandas matplotlib scikit-image
   ```

3. Run the notebook cells in order.
4. At the checkpoint upload prompt, upload both supplied `.pth` files.
5. Allow the notebook to reconstruct the models and load their saved parameters.
6. Upload one or more pet images when prompted.
7. Set `MODEL_TO_DEMO` to `'attention'` or `'base'` to select the model.

The output includes a predicted breed, classification confidence, and a segmentation overlay. Ground-truth labels and masks are required to calculate prediction quality metrics; uploaded images alone only support an inference demonstration.

## Train the models

1. Open `CSE428_Final.ipynb` in Colab. A GPU runtime is useful for training.
2. Install any missing dependencies:

   ```python
   %pip install torch numpy pandas matplotlib scikit-image scikit-learn
   ```

3. Obtain the Oxford-IIIT Pet dataset and upload `images.tar.gz` and `annotations.tar.gz` to `/content/`.
4. Run the extraction cells. The default dataset location is `/content/oxford-iiit-pet`.
5. Check `DATASET_ROOT` and inspect the image/mask overlays before training.
6. Run the remaining cells in order to train and evaluate both models.
7. Download the generated `base_unet_weights.pth` and `attention_unet_weights.pth` files.

For local Jupyter, change `DATASET_ROOT`, extract the dataset locally, and replace Colab-specific upload cells with local file paths.

## Evaluation

The training notebook reports:

- **Segmentation:** binary mean IoU, Dice coefficient, and pixel accuracy.
- **Classification:** accuracy and macro-averaged precision, recall, and F1.
- **Comparison:** results for Base U-Net and Attention U-Net on the test set.

See the saved notebook outputs for the recorded run. Results depend on the dataset, environment, and training run; this README does not claim a separately verified benchmark.

## Scope and limitations

This is an academic prototype trained for pet images and the dataset's breed labels. Behavior on unrelated images or unseen breeds has not been established. The training notebook loads image arrays into memory, so available RAM matters. A complete pinned dependency environment is not supplied.

## Acknowledgements

The project uses the Oxford-IIIT Pet dataset and U-Net-based architectures. Dataset ownership remains with its original creators. The original notebooks are retained as the source of implementation details.

