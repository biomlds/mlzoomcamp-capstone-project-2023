import pandas as pd
import streamlit as st

# from kedro.io import DataCatalog
# from kedro.config import OmegaConfigLoader
import sklearn

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
from yaml import dump

KEDRO_PROJECT_PATH = "/home/kedro/us-insurance/"

st.write("## Insurance charges prediction")

with st.container():
    with st.form("record_form"):
        st.write("### Enter the data:")
        record = {
            "sex": st.selectbox("Sex", ("male", "female"), index=0),
            "region": st.selectbox(
                "Region",
                ("southwest", "southeast", "northwest", "northeast"),
                index=2,
            ),
            "age": st.number_input(
                "Age", min_value=0, value=50, placeholder="Type a number..."
            ),
            "children": st.number_input(
                "Children", min_value=0, value=1, placeholder="Type a number..."
            ),
            "smoker": st.radio("Smoker", ["yes", "no"], key="visibility", index=1),
            "bmi": st.number_input(
                "BMI", min_value=0.0, value=30.97, placeholder="Type a number..."
            ),
        }

        submitted = st.form_submit_button("Predict Charges")

    if submitted:
        # st.write(record)
        with open(Path(KEDRO_PROJECT_PATH, "./conf/local/parameters.yml"), "w") as f:
            dump({"rf_regressor_prediction": record}, f)
        bootstrap_project(Path(KEDRO_PROJECT_PATH))
        with KedroSession.create(project_path=Path(KEDRO_PROJECT_PATH)) as session:
            output_data = session.run(pipeline_name="inference")
        charges = round(output_data["inference_prediction"][0], 2)
        st.write(f"### Predicted value is {charges}")
