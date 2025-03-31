import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit Page Config
st.set_page_config(page_title="Soil Health and Test Case Analyzer", page_icon="🌱", layout="wide")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your soil data CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    # Convert ML column to numeric (handling errors)
    if "ML" in df.columns:
        df["ML"] = df["ML"].astype(str).str.replace("ml", "", regex=True)
        df["ML"] = pd.to_numeric(df["ML"], errors="coerce").fillna(0).astype(int)

    # Ensure required columns exist for Test Case Analysis
    required_cols = {"Soil_Type", "ML", "Test_Case"}
    if required_cols.issubset(df.columns):
        st.subheader("🔍 Test Case Analysis")

        # User Inputs
        soil_types = df["Soil_Type"].unique()
        selected_soil = st.selectbox("🌱 Select Soil Type", soil_types)

        ml_options = sorted(df["ML"].unique())  # Get unique ML values from dataset
        selected_ml = st.selectbox("💧 Select ML Amount", ml_options)

        # Button to trigger prediction
        if st.button("🔍 Predict Best, Moderate, and Worst Cases"):
            # Filter Dataset based on user selection
            filtered_df = df[(df["Soil_Type"] == selected_soil) & (df["ML"] == selected_ml)]

            if filtered_df.empty:
                st.warning("⚠️ No data available for this combination.")
            else:
                st.write(f"📊 **Found {len(filtered_df)} test cases for Soil Type {selected_soil} and ML {selected_ml}ml.**")

                # Train Model
                X = filtered_df.drop(columns=["Test_Case"])  # Features
                X = X.apply(pd.to_numeric, errors="coerce").fillna(0)  # Ensure all numeric

                y = X.mean(axis=1).rank(method="dense").astype(int)  # Rank test cases
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict Best, Moderate, and Worst Cases
                filtered_df["Predicted_Rank"] = model.predict(X)
                sorted_df = filtered_df.sort_values(by="Predicted_Rank")

                # Extract actual data for Best, Moderate, and Worst Cases
                best_case_data = sorted_df.iloc[-1].drop(["Test_Case", "Predicted_Rank"])
                moderate_case_data = sorted_df.iloc[len(sorted_df) // 2].drop(["Test_Case", "Predicted_Rank"])
                worst_case_data = sorted_df.iloc[0].drop(["Test_Case", "Predicted_Rank"])

                # Create a DataFrame for better visualization
                test_case_table = pd.DataFrame({
                    "Test Case Type": ["🏆 Best Case", "⚖️ Moderate Case", "❌ Worst Case"],
                    "Test Case ID": [sorted_df.iloc[-1]["Test_Case"], sorted_df.iloc[len(sorted_df) // 2]["Test_Case"], sorted_df.iloc[0]["Test_Case"]],
                    "Selected ML": [selected_ml] * 3,
                    "Soil Type": [selected_soil] * 3
                })

                # Merge with actual data
                best_case_data = pd.DataFrame([best_case_data], index=["Best Case"])
                moderate_case_data = pd.DataFrame([moderate_case_data], index=["Moderate Case"])
                worst_case_data = pd.DataFrame([worst_case_data], index=["Worst Case"])

                # Combine into one table
                final_table = pd.concat([best_case_data, moderate_case_data, worst_case_data])
                final_table.insert(0, "Test Case Type", test_case_table["Test Case Type"].values)
                final_table.insert(1, "Test Case ID", test_case_table["Test Case ID"].values)
                final_table.insert(2, "Soil Type", test_case_table["Soil Type"].values)
                final_table.insert(3, "Selected ML", test_case_table["Selected ML"].values)

                # Display the table in Streamlit
                st.subheader("📊 Best, Moderate, and Worst Test Cases")
                st.dataframe(final_table)

                # Visualization
                fig = px.bar(
                    sorted_df, x="Test_Case", y="Predicted_Rank",
                    title="Test Case Rankings", labels={"Predicted_Rank": "Score"},
                    color="Predicted_Rank", template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True, key="test_case_chart")

    # 🌈 Spectral Data Visualization (Pie Chart for 410-940 nm Range)
    st.subheader("🌈 Spectral Data Distribution")

    # Identify spectral columns in the 410-940 nm range
    spectral_columns = [col for col in df.columns if col.replace(".", "").isdigit() and 410 <= float(col) <= 940]

    if spectral_columns:
        # Calculate mean spectral values for visualization
        spectral_data = df[spectral_columns].mean()

        # Create Pie Chart
        fig_pie = px.pie(
            names=spectral_data.index,
            values=spectral_data.values,
            title="Spectral Component Distribution (410-940 nm)",
            hole=0.3,  # Donut-style visualization
        )

        # Display the pie chart
        st.plotly_chart(fig_pie, use_container_width=True, key="spectral_pie_chart")
    else:
        st.warning("⚠️ No spectral data columns found in the 410-940 nm range.")

    # Train Model for Soil Health Prediction
    if "Soil Health" in df.columns:
        st.subheader("🔬 Train Model to Predict Soil Health")

        # Drop rows where 'Soil Health' is missing
        df = df.dropna(subset=["Soil Health"])

        features = df.drop(columns=["Soil Health"])  # All except target
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0)  # Ensure numeric

        target = df["Soil Health"].astype(str).str.strip()  # Clean up text

        # Convert categorical target to numeric
        target_map = {"Good": 2, "Moderate": 1, "Poor": 0}
        target = target.map(target_map)

        if target.isna().sum() > 0:
            st.error("⚠️ Soil Health column contains unknown values. Please check your dataset.")
        elif target.empty:
            st.error("⚠️ Not enough valid data to train the model.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.write(f"✅ **Model Accuracy: {accuracy:.2f}**")

            # Prediction Section
            st.subheader("🔮 Predict Soil Health")
            input_values = {col: st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].median())) for col in features.columns}

            if st.button("Predict Soil Health"):
                input_df = pd.DataFrame([input_values])
                prediction = model.predict(input_df)[0]
                predicted_label = [key for key, val in target_map.items() if val == prediction][0]
                st.write(f"🧪 **Predicted Soil Health: {predicted_label}**")
                # 🌱 Dynamic Soil Health Recommendations
                st.subheader("🛠️ Soil Health Improvement Recommendations")

                # Find feature importance to give data-based suggestions
                feature_importance = pd.DataFrame({
                    "Feature": features.columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                # Get top 3 influential factors affecting soil health
                top_factors = feature_importance["Feature"].head(3).tolist()

                if predicted_label == "Moderate":
                    st.markdown("### 🟡 **Moderate Soil Health:**")
                    st.write(f"✅ **Improve {top_factors[0]}:** Increase or balance it as needed.")
                    st.write(f"✅ **Manage {top_factors[1]}:** Ensure its optimal level.")
                    st.write(f"✅ **Adjust {top_factors[2]}:** Small changes may improve overall soil health.")

                elif predicted_label == "Poor":
                    st.markdown("### 🔴 **Poor Soil Health:**")
                    st.write("❌ **Immediate Attention Needed!**")
                    st.write(f"✅ **Improve {top_factors[0]}:** This is the most critical factor affecting soil quality.")
                    st.write(f"✅ **Balance {top_factors[1]} and {top_factors[2]}:** Necessary to prevent further deterioration.")

  

                



else:
    st.info("📂 Please upload a dataset to proceed.")
