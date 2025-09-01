# Heart Disease Prediction App

## Installation

To run the Heart Disease Prediction App, you'll need to have the following dependencies installed:

- Python 3.x
- Streamlit
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using pip:

```
pip install streamlit numpy pandas scikit-learn
```

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/heart-disease-prediction.git
```

2. Navigate to the project directory:

```
cd heart-disease-prediction
```

3. Run the Streamlit app:

```
streamlit run app.py
```

This will launch the Heart Disease Prediction App in your default web browser.

## API

The app uses the following API:

- `streamlit`: For creating the web-based user interface.
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `sklearn.model_selection.train_test_split`: For splitting the dataset into training and testing sets.
- `sklearn.linear_model.LogisticRegression`: For training the logistic regression model.
- `sklearn.metrics.accuracy_score`: For calculating the accuracy of the trained model.
- `sklearn.preprocessing.StandardScaler`: For standardizing the input features.

## Contributing

If you'd like to contribute to the Heart Disease Prediction App, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request to the original repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

To run the tests for the Heart Disease Prediction App, you can use the following command:

```
pytest tests/
```

This will run any tests that have been implemented for the project.
