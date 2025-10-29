# NBA Lineup Optimization: Fifth Player Prediction

## Overview  
This project predicts the optimal fifth player for a home team in NBA games using historical data (2007–2015). It leverages machine learning models to analyze player embeddings, lineup synergies, and game statistics, evaluating performance through accuracy, AUC-ROC, and F1-scores. The best-performing model (KNN with PCA) achieves **71.49% accuracy**, outperforming Logistic Regression, SVM, Neural Networks, and Random Forest.

---

## Key Features  
- **Data Preprocessing**: Handles class imbalance via resampling and engineered features (e.g., turnover percentage, offensive rebound rate).  
- **Player Embeddings**: Encodes players into 32D vectors to capture latent performance traits.  
- **Model Suite**: Tests five models (Logistic Regression, SVM, Neural Network, Random Forest, KNN).  
- **Dimensionality Reduction**: Uses PCA to reduce features from 320D to 253D (95% variance retained).  
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and AUC-ROC.  
- **Actionable Recommendations**: Generates top fifth-player candidates with win probabilities.  

---

## How It Works  

### Workflow Steps  
1. **Data Loading**:  
   - Reads CSV files (e.g., `matchups-2007.csv`) containing lineup and game stats.  
   - Columns: `home_0` to `home_4` (players), game stats (FG%, rebounds, etc.), `outcome` (win/loss).  

2. **Data Preprocessing**:  
   - **Class Balancing**: Upsamples minority class to address imbalance.  
   - **Player Encoding**: Converts player names to IDs using `LabelEncoder`.  
   - **Feature Engineering**: Derives advanced metrics (e.g., `field_goal_efficiency`, `turnover_percentage`).  

3. **Model Training**:  
   - **Embeddings**: Neural network generates 32D player embeddings.  
   - **Dimensionality Reduction**: Applies PCA to embeddings and game stats.  
   - **Algorithms**:  
     - Logistic Regression, SVM, Neural Network, Random Forest, KNN.  

4. **Evaluation**:  
   - Metrics: Accuracy, precision, recall, F1-score, AUC-ROC.  
   - Output: Confusion matrices, ROC curves, classification reports.  

## Requirements  
### Libraries  
- Python 3.7+  
- pandas  
- numpy  
- scikit-learn  
- keras  
- matplotlib  
- seaborn  

### Input Data Format  
CSV files (`matchups-YYYY.csv`) with columns:  
- `game`, `season`, `home_team`, `away_team`  
- `home_0` to `home_4`, `away_0` to `away_4` (player IDs)  
- Game stats: `fga_home`, `reb_home`, `pts_home`, etc.  
- Target: `outcome` (1 = home win, -1 = away win)  

---

## Usage  
1. **Install Dependencies**:  
   ```bash
   pip install numpy pandas scikit-learn keras matplotlib seaborn
2. **Ensure CSV files are in the working directory.**
3. **Run the script**: Use Google Colab or Jupyter Notebook for sequential code compilation and execution.

## Results
5th Player Prediction with Single Highest Probable Player

![Alt text](results/Single_Player_Accuracy.png?raw=true "Single Player Overall Average Accuracy")

![Alt text](results/Single_Player_Prediction.png?raw=true "Single Player Year on Year Accuracy")

5th Player Prediction with 3 Highest Probable Players

![Alt text](results/3_Player_Accuracy.png?raw=true "3 Player Overall Average Accuracy")

![Alt text](results/3_Player_Prediction.png?raw=true "3 Player Year on Year Accuracy")

5th Player Prediction with 5 Highest Probable Players

![Alt text](results/5_Player_Accuracy.png?raw=true "5 Player Overall Average Accuracy")

![Alt text](results/5_Player_Prediction.png?raw=true "5 Player Year on Year Accuracy")
