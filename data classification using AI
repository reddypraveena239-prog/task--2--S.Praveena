# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# Features (input data)
X = iris.data

# Target labels (output)
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create classification model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display output
print("AI Classification Model")
print("Accuracy:", accuracy)

# Test with a sample flower
sample_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample_data)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("Setosa")
elif prediction[0] == 1:
    print("Versicolor")
else:
    print("Virginica")