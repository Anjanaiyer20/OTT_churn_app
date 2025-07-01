import pandas as pd

# Define the dataset
data = {
    "Gender": ["Male", "Female", "Female", "Male", "Male", "Female", "Male", "Female", "Male", "Female",
               "Male", "Female", "Female", "Male", "Male", "Female", "Female", "Male", "Male", "Female"],
    "Age": [25, 32, 29, 40, 22, 28, 35, 31, 45, 26, 38, 24, 30, 41, 27, 36, 33, 23, 34, 39],
    "Subscription_Type": ["Basic", "Premium", "Standard", "Basic", "Standard", "Premium", "Basic", "Standard",
                          "Premium", "Standard", "Basic", "Premium", "Basic", "Standard", "Premium", "Basic",
                          "Premium", "Standard", "Basic", "Premium"],
    "Watch_Time": [0.5, 3.0, 1.5, 0.2, 0.0, 4.5, 0.8, 2.5, 4.8, 1.0, 0.3, 4.0, 0.7, 1.8, 3.9, 0.1, 4.2, 1.1, 0.6, 3.3],
    "Num_Devices": [1, 3, 2, 1, 1, 4, 2, 3, 4, 2, 1, 4, 2, 2, 3, 1, 4, 2, 1, 4],
    "Last_Login_Days": [40, 5, 20, 50, 60, 1, 35, 10, 2, 25, 55, 1, 45, 15, 3, 65, 2, 30, 48, 4],
    "Customer_Support_Calls": [3, 0, 2, 4, 5, 0, 2, 1, 0, 2, 4, 1, 3, 1, 0, 5, 1, 2, 4, 0],
    "Churn": [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save it to a CSV file
df.to_csv("ott_churn_data.csv", index=False)

print("✅ Dataset saved as 'ott_churn_data.csv'")
