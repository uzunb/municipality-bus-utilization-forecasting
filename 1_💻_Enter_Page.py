import streamlit as st
from streamlit.logger import get_logger
import streamlit.components.v1 as components
import datetime

LOGGER = get_logger(__name__)

thedate = datetime.date.today()


def main():
    st.set_page_config(page_title="Enter Page", page_icon="💻")
    st.image(r'./resources/bus_enter_page.png', use_column_width=True)


    st.write("""
    # Welcome :clap:
    """)

    st.markdown(
        """
    This repo was developed for the P.I. Works Technical Assignment.
    
    ### Goal
    The goal of this project is to forecast bus demands of municipalities in Banana Republic using the provided by the dataset.

    """
    )
    st.markdown("""
                ------
                ###### Machine Learning Model Deployment
                ###### Version: 1.0
                ###### Date: {}
                ![Visitor count](https://shields-io-visitor-counter.herokuapp.com/badge?page=https://share.streamlit.io/your_deployed_app_link&label=VisitorsCount&labelColor=000000&logo=GitHub&logoColor=FFFFFF&color=1D70B8&style=for-the-badge)
                """.format(str(thedate)))


if __name__ == "__main__":
    main()
