# House Price Prediction ML Project

## Project Overview
This is a Machine Learning project that predicts house prices using the California Housing Dataset. Built with Python, scikit-learn, and Streamlit for a web interface.

## Objective
- Predict house prices based on features like income, house age, rooms, etc.
- Compare Linear Regression vs Random Forest models
- Deploy as a web application

## Dataset
- Source: California Housing Dataset (sklearn)
- Rows: 20,640
- Features: 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- Target: Price (median house value in 100k units)

## Technologies Used
- Python 3.x
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine Learning
- Streamlit - Web framework
- matplotlib, seaborn - Visualization

## Project Structure

house-price-prediction/
├── data/
│   ├── house_prices.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
├── models/
│   └── best_model.pkl
├── download_dataset.py
├── 01_data_exploration.py
├── 02_data_preprocessing.py
├── 03_model_training.py
├── app.py
├── .gitignore
└── README.md

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip

### Steps
1. Clone the repository:
git clone https://github.com/faizazam23/house-price-prediction.git
cd house-price-prediction

2. Create virtual environment:
python -m venv venv

3. Activate virtual environment:
   Windows:
venv\Scripts\activate
   Mac/Linux:
source venv/bin/activate

4. Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

5. Run the web app:
streamlit run app.py

## Model Performance
| Model | Training R² | Testing R² | RMSE |
|-------|-------------|-----------|------|
| Linear Regression | 0.6126 | 0.5758 | 0.7456 |
| Random Forest | 0.9735 | 0.8046 | 0.5060 |

Best Model: Random Forest 

## How to Use
1. Launch the app: streamlit run app.py
2. Adjust sliders in the sidebar for house features
3. See real-time price prediction
4. Check model performance metrics

## Project Workflow

### 1. Data Exploration (01_data_exploration.py)
- Load and explore dataset
- Check statistics and distributions
- Identify feature correlations

### 2. Data Preprocessing (02_data_preprocessing.py)
- Handle missing values
- Split data (80% train, 20% test)
- Save processed data

### 3. Model Training (03_model_training.py)
- Train Linear Regression model
- Train Random Forest model
- Compare and select best model

### 4. Web Application (app.py)
- Build Streamlit interface
- Interactive prediction tool
- Display model performance

## Key Features
- Interactive sliders for predictions
- Real-time price calculation
- Model performance metrics
- Clean and intuitive UI
- Feature importance analysis

## Future Improvements
- Add more advanced models (XGBoost, LightGBM)
- Feature engineering and scaling
- Cross-validation for better accuracy
- Deploy to cloud (Heroku, AWS)
- Add more visualizations

## Author
- Name: Faiyz Azam
- Email: faiyz.azam@gmail.com
- GitHub: @faizazam23

## License
MIT License - feel free to use this project!

## Acknowledgments
- California Housing Dataset (sklearn)
- Streamlit documentation
- Machine Learning community

Happy Predicting! 🚀
