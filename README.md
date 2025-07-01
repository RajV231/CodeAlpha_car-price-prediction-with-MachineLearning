# CodeAlpha_car-price-prediction-with-MachineLearning

This project predicts the **selling price of used cars** based on various features like brand, fuel type, age, and more using regression models.

##  Objective
To demonstrate the use of machine learning for real-world price prediction tasks in the automotive resale market.

---

##  Dataset

The dataset includes:

- Car Name
- Year of Purchase
- Present Price
- Kilometers Driven
- Fuel Type
- Selling Type (Dealer/Individual)
- Transmission (Manual/Automatic)
- Number of Previous Owners
- Selling Price (target)

---

##  Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn / Matplotlib
- Jupyter Notebook

---

##  Workflow

1. **Data Cleaning**: Drop irrelevant columns, handle categorical data.
2. **Feature Engineering**: Create `Car Age`, drop `Car Name`, convert strings to numerics.
3. **Model Training**:
   - Linear Regression
   - Random Forest Regressor
4. **Evaluation**:
   - R² Score
   - RMSE
   - Visualization: Actual vs Predicted Plot

---

##  Results

- **Linear Regression R²**: ~0.85  
- **Random Forest R²**: ~0.96  
- **Random Forest RMSE**: ~0.97  

---

##  Output Example

![Prediction Plot](assets/prediction_plot.png) *(Optional: Add this if you save the plot)*

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

##  Author

Rajvardhan Wakharade  
_Data Science Intern_

---

