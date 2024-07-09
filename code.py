import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Function to load food product data from Excel file
def load_food_product_data(file_path):
    df = pd.read_excel(file_path)
    return df


def combine_ingredients(row):
    ingredients = ' '.join(row.dropna().drop('Prediction').values)
    unique_words = set(ingredients.replace(',', ' ').split())
    return list(unique_words)


def main():
    # Load food product data
    df = pd.read_excel(r"C:\Users\tanya\Desktop\cleanDataReutTanya.xlsx")

    # Define allergies
    allergies = {
        'gluten_allergy': ['pasta', 'pizza', 'wheat', 'bread', 'crackers', 'ladyfingers', 'dough', 'semolina', 'flour'],
        'lactose_allergy': ['cheese', 'butter', 'yogurt', 'cottage', 'milk', 'brie', 'cheese', 'cream'],
        'G6PD_allergy': ['lentils', 'black beans', 'chickpeas', 'kidney beans', 'soybeans'],
        'Nuts': ['almonds', 'peanuts', 'pecans', 'peanuts'],
        'vegan': ['chicken', 'beef', 'lamb', 'paneer', 'bacon', 'fish', 'salmon', 'cod', 'tuna', 'pork', 'ribs',
                  'sausage', 'shrimp', 'prawns', 'lobster', 'eggs']
    }

    # Extract ingredients
    ingredients = df['all_ingredients'].apply(lambda x: ','.join(eval(x)))

    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(ingredients)

    # Define multiple target variables (allergies)
    y_gluten = df['gluten_allergy']
    y_lactose = df['lactose_allergy']
    y_G6PD = df['G6PD_allergy']
    y_vegan = df['vegan']

    # Split data into train and test sets
    X_train, X_test, y_gluten_train, y_gluten_test = train_test_split(X, y_gluten, test_size=0.2, random_state=42)
    _, _, y_lactose_train, y_lactose_test = train_test_split(X, y_lactose, test_size=0.2, random_state=42)
    _, _, y_G6PD_train, y_G6PD_test = train_test_split(X, y_G6PD, test_size=0.2, random_state=42)
    _, _, y_vegan_train, y_vegan_test = train_test_split(X, y_vegan, test_size=0.2, random_state=42)

    # Initialize classifiers
    clf_gluten = MultinomialNB()
    clf_lactose = MultinomialNB()
    clf_G6PD = MultinomialNB()
    clf_vegan = MultinomialNB()

    # Train classifiers
    clf_gluten.fit(X_train, y_gluten_train)
    clf_lactose.fit(X_train, y_lactose_train)
    clf_G6PD.fit(X_train, y_G6PD_train)
    clf_vegan.fit(X_train, y_vegan_train)

    # Make predictions
    y_gluten_pred = clf_gluten.predict(X_test)
    y_lactose_pred = clf_lactose.predict(X_test)
    y_G6PD_pred = clf_G6PD.predict(X_test)
    y_vegan_pred = clf_vegan.predict(X_test)

    # Calculate accuracy scores
    accuracy_gluten = accuracy_score(y_gluten_test, y_gluten_pred)
    accuracy_lactose = accuracy_score(y_lactose_test, y_lactose_pred)
    accuracy_G6PD = accuracy_score(y_G6PD_test, y_G6PD_pred)
    accuracy_vegan = accuracy_score(y_vegan_test, y_vegan_pred)

    #  accuracy
    # print(f"Accuracy - Gluten Allergy: {accuracy_gluten:.2f}")
    # print(f"Accuracy - Lactose Allergy: {accuracy_lactose:.2f}")
    # print(f"Accuracy - G6PD Allergy: {accuracy_G6PD:.2f}")
    # print(f"Accuracy - Vegan: {accuracy_vegan:.2f}")

    # Generate confusion matrixes
    def plot_confusion_matrix(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap='Blues', values_format='.0f')
        plt.title(title)
        plt.show()

   #the matrix plotting
    # plot_confusion_matrix(y_gluten_test, y_gluten_pred, "Confusion Matrix - Gluten Allergy")
    # plot_confusion_matrix(y_lactose_test, y_lactose_pred, "Confusion Matrix - Lactose Allergy")
    # plot_confusion_matrix(y_G6PD_test, y_G6PD_pred, "Confusion Matrix - G6PD Allergy")
    # plot_confusion_matrix(y_vegan_test, y_vegan_pred, "Confusion Matrix - Vegan")

    # Function to check if a food is safe to eat for a given allergy
    def check_food_safety(food, allergy):
        # Vectorize the input food
        food_vector = vectorizer.transform([food])

        # Determine which classifier to use based on the allergy
        if allergy == 'gluten_allergy':
            clf = clf_gluten
        elif allergy == 'lactose_allergy':
            clf = clf_lactose
        elif allergy == 'G6PD_allergy':
            clf = clf_G6PD
        elif allergy == 'vegan':
            clf = clf_vegan
        else:
            print(f"Unknown allergy: {allergy}")
            return False

        # Predict if the food is safe (0 = safe, 1 = not safe)
        prediction = clf.predict(food_vector)[0]

        if prediction == 0:
            return f"You can eat '{food}'. No problematic ingredients were found."
        else:
            return f"Warning: '{food}' may not be safe for your {allergy}."

    # Example 
    food = input("Enter a food: ").strip().lower()
    allergy = input("Enter an allergy (gluten_allergy, lactose_allergy, G6PD_allergy, vegan): ").strip().lower()

    if allergy in allergies:
        result = check_food_safety(food, allergy)
        print(result)
    else:
        print(f"Unknown allergy: {allergy}. Please enter a valid allergy type.")


if __name__ == "__main__":
    main()
