# Adaptive Gait Rehabilitation using Reinforcement Learning & Algae Method

## 🧠 Project Overview

Gait rehabilitation is a vital area of medical research, dedicated to restoring and enhancing mobility for people with lower-limb impairments. While conventional aids like exoskeletons offer support, they are limited by their need for manual tuning and their inability to truly adapt to a user's changing needs. **Reinforcement Learning (RL)** offers a powerful solution, enabling the creation of intelligent control systems that can make real-time, context-aware adjustments during rehabilitation.

This project advances that capability by presenting a **hybrid RL framework**. It utilizes a **Convolutional Neural Network (CNN)** as its policy backbone, trains it using **Proximal Policy Optimization (PPO)** principles, and, crucially, incorporates a population-based genetic approach (the **Algae Method**). This integration is designed to build a highly adaptable and robust RL system that can learn from and tailor control strategies to the unique requirements of each patient, promising significantly enhanced gait rehabilitation outcomes.

## 🚀 Key Features
* **AlgaePPO Framework:** Population-based reinforcement learning strategy with **evolutionary principles** (crossover and mutation) for policy optimization.
* **Custom CNN Policy:** A deep learning model designed to process $128 \times 128$ grayscale images of gait data.
* **Interpretability with SHAP:** Utilizes **SHAP (SHapley Additive exPlanations)** to explain the model's predictions, crucial for trustworthy application in a medical domain.
* **Modular Design:** Separated components for data handling, model architecture, training, and evaluation.

## ⚙️ Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | Main execution script for training, plotting, saving, and evaluating the AlgaePPO agent. |
| `algae_ppo.py` | Implementation of the `AlgaePPO` class, managing the population of policies, training, and evolutionary selection. |
| `cnn_model.py` | Definition of the **Convolutional Neural Network (CNN)** architecture. |
| `dataset.py` & `image_loader.py` | Classes and functions for loading, transforming, and normalizing image data for training. |
| `shap_explain.py` | Utility for generating and visualizing **SHAP values** for model interpretability. |
| `train_evaluate.py` | Helper functions for advanced evaluation metrics, including **ROC/AUC curve plotting** and standard training loops (currently unused in `main.py` but available for future expansion). |

## 🛠️ Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/siddharth-haveliwala/Adaptive-Gait-Rehabilitation-using-Reinforcement-Learning-with-Algae-Optimization.git
    cd Adaptive-Gait-Rehabilitation-AlgaePPO
    ```
2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:** (Create a `requirements.txt` based on the code)
    ```bash
    pip install torch torchvision numpy pandas matplotlib scikit-learn shap pillow
    ```
4.  **Data Setup:**
    Create a `data/` directory and structure your gait image data as follows:
    ```
    data/
    ├── 40_Percent/
    │   ├── img_001.png
    │   └── ...
    ├── 70_Percent/
    │   ├── img_001.png
    │   └── ...
    └── ... (for 80_Percent, 100_Percent)
    ```

## ▶️ How to Run

1.  **Methodology:**

    - **Data Loading and Preprocessing**

      - **Datasets:** We utilized the OU-Biometrics dataset (OU-ISIR), specifically focusing on the TreadmillDatasetB and TreadmillDatasetD.
      - **Preprocessing:** Input images were converted to tensors and normalized to ensure consistency when feeding data into the neural network.

    - **Convolutional Neural Network (CNN)**

      - **Architecture:** The model consists of convolutional layers, fully connected layers, and an output layer designed for the specific actions relevant to the dataset.
      - **Dataset B:** Trained on a 3-layer CNN with ReLU activation, batch normalization, and max-pooling.
      - **Dataset D:** Trained to analyze NAC (Normalized AutoCorrelation) values to detect gait abnormalities.

    - **Proximal Policy Optimization (PPO) + Algae Optimization**

      - **PPO:** Balances exploration and exploitation using a clipping mechanism to stabilize policy updates.
      - **Algae Optimization:** Genetic algorithm-inspired method involving crossover and mutation operations to evolve neural network weights, enhancing robustness and adaptability.

    - **Training and Testing**

      - **Training Episodes:** 50 episodes for Dataset B and 10 episodes for Dataset D, considering data volume and complexity.
      - **Evaluation:** Models were evaluated for fitness, with weights updated based on performance metrics like loss, optimizers, and evolution parameters.
      - **Crossover and Mutation:** These genetic operations were applied every 5 episodes to introduce variability and evolve the model population.
  
  2.  **Training and Evaluation:**
    Execute the main script. This will train the 3 models for 50 episodes, plot the accuracy, save the best model, and run a final evaluation.
    ```bash python main.py```
   
## 📊 Results and Interpretability

The training process is logged to the console, and a `Accuracy vs Episodes` plot is generated.

#### Model Interpretability (SHAP)
After evaluation, the `shap_explain.py` script can be executed (requires slight modification to integrate with `main.py`) to generate visual explanations. 
*Visualization of SHAP values overlaid on a gait image, showing regions that positively (red) and negatively (blue) contributed to the classification. This is critical for medical application verification.*

#### Dataset B

- **Training Loss:** Observed fluctuating training loss, but algae optimization effectively evolved better models.
- **Test Accuracy:** Achieved high accuracy (above 99%) after 50 episodes, demonstrating robust gait pattern segmentation.

### Dataset

- **NAC Values:** The optimal shift value was found to be 69, with an NAC value close to 0.975, indicating a stable and consistent gait cycle.
- **Test Accuracy:** Consistently improving test accuracy across epochs, reaching perfect accuracy, with SHAP analysis providing insights into model decision-making.

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## References

1. [The OU-ISIR Gait Database Comprising the Treadmill Dataset](https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/)
2. [Reward-Adaptive Reinforcement Learning: Dynamic Policy Gradient Optimization for Bipedal Locomotion](https://arxiv.org/pdf/2107.01908.pdf)
3. [A Data-Driven Reinforcement Learning Solution Framework for Optimal and Adaptive Personalization of a Hip Exoskeleton](https://arxiv.org/ftp/arxiv/papers/2011/2011.06116.pdf)
4. [Static Standing Balance with Musculoskeletal Models Using PPO With Reward Shaping](https://tinyurl.com/mvktmf7z)
5. [A Novel Deep Reinforcement Learning Based Framework for Gait Adjustment](https://arxiv.org/pdf/2107.01908.pdf)
6. [Challenges with Reinforcement Learning in Prosthesis](https://www.mdpi.com/2227-7390/11/1/178)
