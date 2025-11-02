import streamlit as st
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TransparencyAnalyzer:
    """
    Handles generating explanations for models using LIME.
    SHAP is handled directly in the streamlit app for its interactive plots.
    """

    def explain_with_lime(self, model, X_test_df, instance_idx):
        """
        Generates a LIME explanation for a specific instance.
        
        Args:
            model (sklearn.base.BaseEstimator): The trained classifier.
            X_test_df (pd.DataFrame): The test features (transformed).
            instance_idx (int): The index of the instance to explain.
            
        Returns:
            matplotlib.figure.Figure: The LIME explanation plot.
        """
        
        # LIME needs the training data statistics to generate perturbations
        # We use X_test_df as a proxy for the training data distribution
        
        # Get class names from the model
        class_names = model.classes_
        
        # Create a LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_test_df.values,
            feature_names=X_test_df.columns.tolist(),
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        # Get the instance to explain
        instance = X_test_df.iloc[instance_idx]
        
        # Generate the explanation
        # model.predict_proba is the function LIME needs
        explanation = explainer.explain_instance(
            data_row=instance.values,
            predict_fn=model.predict_proba,
            num_features=10  # Show top 10 features
        )
        
        # Get the figure from the explanation
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        return fig