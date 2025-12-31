import streamlit as st
import altair as alt
import pandas as pd

# Handle both direct execution and package import
try:
    from .ahp_computation import BestEncoder
except ImportError:
    from ahp_computation import BestEncoder

st.set_page_config(page_title="EncodeHero")

class ComparisonApp:
    def __init__(self):

        # Define criteria groups
        self.criteria_groups = {
            'Criteria': ['Performance', 'Cost'],
            'Performance': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC'],
            'Cost': ['Time']
        }
        self.comparisons = {group: {} for group in self.criteria_groups}
        self.temp_comparisons = {group: {} for group in self.criteria_groups}
    
    def ranking_plot(self, res: dict):
        """
        Generate and display a bar chart ranking encoders based on their priority values.

        Parameters:
        - res (dict): A dictionary where keys represent encoder names and values represent their corresponding priority values.

        Returns:
        - A bar chart visualizing the ranking of encoders, displayed using Streamlit's `altair_chart` component.
        """
        sorted_items = sorted(res.items(), key=lambda x: x[1])
        data = pd.DataFrame(sorted_items, columns=['Encoder', 'Priority value'])
        
        # Calculate dynamic height based on number of encoders (minimum 20 pixels per bar)
        num_encoders = len(data)
        chart_height = max(400, num_encoders * 20)
        
        chart = alt.Chart(data).mark_bar().encode(
            x='Priority value:Q',
            y=alt.Y('Encoder:N', sort='-x')
        ).properties(
            width=600,
            height=chart_height
        )
        
        st.header('Priority of each encoder')
        st.markdown('''The highest priority value is associated with the most suitable encoder based on your criteria (enlarge the plot to see all encoders).''')
        st.altair_chart(chart, width='stretch')
    
    def consistency_ratio(self):
        """
        Display the consistency ratio for various criteria categories in the AHP (Analytic Hierarchy Process) report.

        Parameters:
        - None (uses `self.report` which contains the consistency ratio data for each category).

        Returns:
        - Displays a message in the Streamlit app regarding the consistency ratio for each of the following categories:
        'Criteria' and 'Performance' (Cost is omitted as it has only one criterion).
        """
        report = self.report

        for key in ["Criteria", "Performance"]:
            ratio = report[key]["elements"]["consistency_ratio"]
            if ratio > 0.1:
                st.write(f"Consistency ratio for {key}: {ratio*100:.2f}%. Try to be consistent with your choices")
            else:
                st.write(f"Consistency ratio for {key}: {ratio*100:.2f}%")

    def global_weights(self):
        """
        Display the global weights for each criterion in the AHP (Analytic Hierarchy Process) report.

        Parameters:
        - None (uses `self.report` which contains the global weights data for 'Criteria' and 'Performance').

        Returns:
        - Displays the global weights of each criterion in a Streamlit app, along with a descriptive header and markdown.
        """
        global_weights = {key: self.report[key]["elements"]["global_weights"] for key in ["Criteria", "Performance"]}
        
        st.header('Global weight of each criteria')
        st.markdown('''Each criterion has an associated weight which indicates its influence on the encoder choice.''')
        st.write(global_weights)


    def run(self):
        """
        Main method to run the AHP encoder comparison tool within a Streamlit app.

        Steps:
        1. Configure the Streamlit app page.
        2. Display a welcome message and guide for the user to make encoder comparisons.
        3. Allow the user to select the task type (binary, multiclass, or regression) and model to analyze.
        4. Provide interactive widgets to allow users to compare different criteria within selected groups.
        5. Once comparisons are made, display the calculated results including consistency ratio, ranking plot, and global weights.

        Parameters:
        - None (uses instance attributes and Streamlit widgets to interact with user inputs).

        Returns:
        - Displays the AHP tool interface and results within a Streamlit app.
        """

        st.title('EncodeHero')

        st.markdown('''Welcome to EncodeHero, the AHP Encoders Pairwise Comparison Tool.

Please make encoders comparisons for the following criteria groups:
- Performance metrics
- Computational cost (Time)

1. Select the preferred task
2. Select the model results to analyze (set *all* to display results over all models)
3. Select the criteria you prefer from the menu
4. Select a value from 1 (equal importance) to 9 (extremely more important)
   for each comparison.
5. Click *Calculate* to compute the results based on your inputs.
                
Enjoy the best encoder for you!''')

        task_selection = st.selectbox(
                            "Select the task",
                            ("binary", "multiclass", "regression"),
                            key="task_selection")
        
        if task_selection == "regression":
            models = ("all", "RF", "MLP", "K-NN", "SVM", "LNR", "DT")
        else:
            models = ("all", "nb", "RF", "MLP", "knn", "SVM", "lgr", "DT")
        
        model_selection = st.radio(
                                "Select the model",
                                models,
                                key="model_selection",
                                horizontal = True)
        
        if task_selection == "regression":
            self.criteria_groups["Performance"] = ['mae', 'mse', 'rmse', 'r2']

        col1, col2 = st.columns([1, 3])

        for group, criteria in self.criteria_groups.items():
            for i, crit1 in enumerate(criteria):
                for j, crit2 in enumerate(criteria):
                    if i < j:
                        var_name_selectbox = f"{crit1}_{crit2}_selectbox"
                        var_name_radio = f"{crit1}_{crit2}_radio"

                        with col1:
                            selectbox_value = st.selectbox(
                                f"{crit1} vs {crit2}",
                                (crit1, crit2),
                                key=f"{var_name_selectbox}_key")
                        with col2:
                            radio_value = st.radio(
                                "Select the importance",
                                list(range(1,10)),
                                key=f"{var_name_radio}_key",
                                horizontal = True,
                            )

                        self.temp_comparisons[group][(crit1, crit2)] = {
                            "selected_criteria" : selectbox_value,
                            "value" : radio_value}

        if st.button("Calculate"):
            self.comparisons = self.temp_comparisons.copy()
            self.report = BestEncoder.get_ranking(task=task_selection, model=model_selection, comparisons=self.comparisons)
            self.consistency_ratio()
            self.ranking_plot(self.report['Criteria']['target_weights'])
            self.global_weights()
        
        # Add footnote
        st.markdown("---")
        st.caption('EncodeHero is part of the *"Categorical Variable Encoding Methods for Tabular Data: A Benchmarking Study"* paper by F. Clerici and N. Nobani')

app = ComparisonApp()
app.run()