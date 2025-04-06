# YOLOv8 Object Detection with Hybrid PSO-GWO Optimization

This project presents a customized implementation of YOLOv8-based object detection, inspired by [entbappy's YOLOv8 Object Detection repository](https://github.com/entbappy/YOLO-v8-Object-Detection). In addition to the base implementation, this project incorporates a **hybrid Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO)** approach to enhance compilation efficiency and reduce training costs.

## Project Highlights

- Implemented object detection using YOLOv8
- Trained the model on a custom dataset using the Ultralytics YOLOv8 framework
- Performed validation and visualization of predictions
- Introduced a hybrid PSO-GWO algorithm for:
  - Efficient hyperparameter tuning
  - Faster convergence during training
  - Reduction in computational resource usage

## Technologies Used

- Python 3.x
- YOLOv8 (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- OpenCV
- PyTorch
- Matplotlib
- Pandas
- Jupyter Notebook

## Project Structure

```
├── Yolov8_object_detection.ipynb       # Main notebook with training and inference code
├── data/                               # Custom dataset directory (YOLO format)
├── README.md                           # Project documentation
```

## Dataset

The model was trained on a custom object detection dataset. You can either:
- Use your own dataset in YOLO format (images and labels)
- Import a dataset using Roboflow for easier preprocessing and formatting

Details about the dataset used (such as number of classes, sample size, and types of objects detected) should be specified based on your implementation.

## Hybrid PSO-GWO Optimization

This project introduces a hybrid optimization approach that combines:

- **Particle Swarm Optimization (PSO)** for robust global exploration of the hyperparameter space
- **Grey Wolf Optimizer (GWO)** for efficient local exploitation during convergence

This hybrid technique was used to optimize parameters such as learning rate, batch size, and number of epochs, resulting in:
- Improved training efficiency
- Reduced computational cost
- Faster convergence compared to standard training methods

## Results and Evaluation

The notebook includes:
- Visualization of training and validation loss
- Sample predictions with bounding boxes
- Model evaluation metrics

These outputs demonstrate the effectiveness of both YOLOv8 and the hybrid optimization strategy.
