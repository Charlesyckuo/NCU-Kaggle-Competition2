# Kaggle Competition 2 - 113522042

This repository contains the code and data for a Kaggle competition participated during a master's course. The project focuses on sequence data analysis, including training a model to make predictions based on time series data. Below, you will find details on the project structure, requirements, and instructions for running the code.

---

## Project Structure

├── train_data.csv # Training dataset │ 
├── test_data.csv # Testing dataset │ 
├── sample_submission.csv # Sample submission format │ 
├── sequence_lengths.csv # Generated time sequence lengths │ 
├── submission.csv # Final model predictions │ 
├── final.py # Main script for training and prediction │ 
├── time_order_mean.py # Script to analyze time sequence statistics
---

## Requirements

The project environment can be set up using the provided `kaggle2.yml` file. Below are the key dependencies:

- **torch**: 2.5.1
- **torchvision**: 0.20.1
- **pandas**: 2.2.2
- **numpy**: 1.26.4
- **scikit-learn**: 1.5.1

---

## Instructions for Running the Code

### 1. Analyze Time Sequence Statistics

To analyze the time sequence lengths in the dataset:

1. Run the script `time_order_mean.py`.
2. Outputs key statistical measures of the sequence lengths, such as:
   - Mean: **2495.300766**
   - Standard Deviation: **389.603178**
   - Minimum: **623**
   - Maximum: **4696**
   - Quartiles: **2287.75 (25%)**, **2553.5 (50%)**, **2756 (75%)**
3. Modify the `SEQ_LEN` parameter in `final.py` based on these insights.
4. The results are saved in `sequence_lengths.csv`, which contains the time steps for each ID.

### 2. Train and Predict

To train the model and make predictions:

1. Run the script `final.py`.
2. Adjust the `EPOCHS` parameter in the script. Recommended value: **110**.
3. Training is GPU-accelerated for high speed.
4. After training, the model generates predictions saved as `submission.csv`.
5. Submit `submission.csv` to Kaggle for evaluation.

---

## Notes

1. **GPU Compatibility**: Ensure your system's CUDA version is compatible with the installed PyTorch version.
2. **Environment Setup**: If using Conda, the environment can be directly initialized with `kaggle2.yml`.

---

## Contact

If you have any questions or encounter issues, feel free to open an issue in this repository.
