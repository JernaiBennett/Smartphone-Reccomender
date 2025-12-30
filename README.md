# Mobile Match: Smart Predictor for Smartphone Prices

A machine learning-powered smartphone price prediction system that helps consumers make informed purchasing decisions by analyzing specification-price relationships across global markets (USA, China, and India).

## 🎯 Project Overview

Built an intelligent price prediction system using multiple regression algorithms to predict smartphone prices based on technical specifications. The system analyzes over 1,100 smartphone models from 18+ brands, achieving 83-91% prediction accuracy across different markets.

**Problem Solved:** Consumers often struggle to determine fair pricing for smartphones with complex specifications. Mobile Match eliminates confusion by predicting prices based on features and identifying the key value drivers in each market.

## 🛠️ Technologies Used

- **Python 3.10** - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical operations
- **Scikit-learn** - Machine learning models and preprocessing
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📊 Machine Learning Models

| Model | Use Case | Best Performance |
|-------|----------|------------------|
| **Linear Regression** | Baseline model for comparison | R² = 0.64-0.82 |
| **Random Forest** | Ensemble learning with decision trees | R² = 0.89 (USA Market) |
| **XGBoost** | Gradient boosting for complex patterns | R² = 0.91 (China Market) |

## ✨ Key Features

### 1. **Multi-Dataset Integration**
- Merged two comprehensive smartphone datasets (980 + 930 models)
- Standardized features across different data sources
- Created unified dataset with 1,109 smartphone models from 18 brands

### 2. **Advanced Data Preprocessing**
- **Feature Extraction**: Cleaned specifications with mixed units (GB, mAh, MP, inches)
- **Label Encoding**: Transformed categorical variables (brand, processor, OS)
- **Feature Engineering**: Created composite features for display quality and computing power
- **Temporal Filtering**: Focused on phones from 2020+ for market relevance

### 3. **Market-Specific Price Prediction**
- **USA Market**: Random Forest (R² = 0.89, RMSE = $148.11)
- **China Market**: XGBoost (R² = 0.91, RMSE = ¥783.27)
- **India Market**: XGBoost (R² = 0.89, RMSE = ₹13,996.38)
- **Global Market**: Random Forest (R² = 0.83)

### 4. **Feature Importance Analysis**
Identified key price drivers across different markets:

**USA Market:**
- Processor Name: 45%
- Brand Name: 17%
- Screen Size: 13%

**China Market:**
- Brand Name: 60%
- Processor Name: 29%

**India Market:**
- Brand Name: 57%
- Processor Name: 32%

### 5. **Comprehensive Visualizations**
- Actual vs. Predicted price scatter plots
- Feature importance rankings
- Cross-model performance comparisons
- Market-specific analysis charts

## 🎓 What I Learned

- **Machine Learning Pipeline**: Designed end-to-end ML workflow from data preprocessing to model deployment
- **Feature Engineering**: Created meaningful features from raw technical specifications
- **Model Selection**: Compared multiple algorithms to identify best-fit models for each market
- **Cross-Market Analysis**: Analyzed consumer preferences and pricing dynamics across regions
- **Data Cleaning**: Handled messy real-world data with inconsistent units and formats
- **Model Evaluation**: Used R², RMSE, and MAE metrics to assess prediction accuracy
- **Ensemble Methods**: Applied Random Forest and XGBoost for robust predictions

## 📁 Project Structure
```
mobile-match/
├── mobile-match.ipynb              # Main Jupyter notebook with full analysis
├── Mobile_Match_Final_Report.pdf   # Comprehensive project report
├── data/
│   ├── smartphones.csv             # Dataset 1 (980 models, 22 features)
│   ├── Mobiles Dataset (2025).csv  # Dataset 2 (930 models, 15 features)
│   └── combined_mobile_data.csv    # Merged dataset (1,109 models)
├── visualizations/
│   ├── feature_importance_*.png
│   ├── model_comparison_*.png
│   └── actual_vs_predicted_*.png
└── README.md
```

## 💻 How to Run

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab
- Required libraries (see requirements below)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mobile-match.git
cd mobile-match
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook mobile-match.ipynb
```

5. **Run all cells** to reproduce the analysis

### Requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

## 📈 Key Results

### Model Performance Summary

| Market | Best Model | R² Score | RMSE | Sample Size |
|--------|-----------|----------|------|-------------|
| USA | Random Forest | 0.89 | $148.11 | 847 |
| China | XGBoost | 0.91 | ¥783.27 | 846 |
| India | XGBoost | 0.89 | ₹13,996.38 | 847 |
| Global | Random Forest | 0.83 | $18,570.76 | 262 |

### Key Insights

1. **Non-linear Pricing**: Tree-based models significantly outperformed Linear Regression, confirming complex pricing relationships
2. **Market Differences**: Different features drive pricing in each market (performance in USA vs. brand in China)
3. **Brand Premium**: Brand name contributes 16-60% of price determination depending on market
4. **Technical Specifications**: RAM and screen size are globally important, accounting for ~50% of price variance

## 🔍 Dataset Information

### Primary Dataset (smartphones.csv)
- **Records**: 980 smartphone models
- **Features**: 22 attributes including 5G support, processor, camera specs, battery capacity
- **Source**: Kaggle - Smartphone Dataset

### Secondary Dataset (Mobiles Dataset 2025.csv)
- **Records**: 930 smartphone models  
- **Features**: 15 attributes with multi-market pricing (USA, China, India, Pakistan, Dubai)
- **Source**: Kaggle - Mobiles Dataset 2025

### Combined Dataset
- **Total Models**: 1,109 unique smartphones
- **Brands**: 18 (Samsung, Apple, Xiaomi, Oppo, Vivo, OnePlus, etc.)
- **Time Range**: 2014-2025 (filtered to 2020+ for analysis)
- **Markets**: USA, China, India with region-specific pricing

## 🎯 Skills Demonstrated

- **Data Science**: Exploratory Data Analysis, statistical analysis, feature engineering
- **Machine Learning**: Regression modeling, hyperparameter tuning, model evaluation
- **Data Preprocessing**: Cleaning messy data, handling missing values, encoding categorical variables
- **Python Programming**: Pandas, NumPy, Scikit-learn, XGBoost
- **Data Visualization**: Creating informative charts with Matplotlib and Seaborn
- **Research & Documentation**: Comprehensive project report with methodology and findings
- **Critical Thinking**: Analyzing market-specific consumer preferences and pricing strategies

## 🚀 Future Enhancements

- **Deep Learning**: Implement neural networks for improved accuracy on premium devices
- **User Interface**: Build web application for interactive price predictions
- **Real-time Data**: Integrate live pricing data from e-commerce platforms
- **Recommendation System**: Suggest best-value smartphones based on user preferences
- **Time Series Analysis**: Predict price depreciation over time
- **Sentiment Analysis**: Incorporate user review sentiment into pricing model

## 📚 Research Paper

View the comprehensive [Mobile Match Final Report](Mobile_Match_Final_Report.pdf) for:
- Detailed methodology
- Statistical analysis
- Feature importance interpretation
- Market-specific insights
- Visualizations and results

## 🙏 Acknowledgments

- **Dataset Sources**: Kaggle contributors (Abdul Malik, Baloch M., Rohit)
- **Institution**: Florida International University
- **Course**: Machine Learning / Data Mining course
- **Collaborator**: Luis Fabrizio Barrera Inga

## 📄 License

This project is open source and available for educational purposes.

---

⭐ **If you found this project helpful, please give it a star!**

**Built with ❤️ by Jernai Bennett** | Undergraduate Computer Science Student at FIU | Class of December 2025
