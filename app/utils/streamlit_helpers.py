# -*- coding: utf-8 -*-
""" Helpers related to streamlit frontend """
import streamlit as st

def init_var(var_list):
    """ Initialising list of key, val in session state if not there """
    for key_, val_ in [(key, val) for key, val in var_list if key not in st.session_state]:
        st.session_state[key_] = val_

def on_change_refresh_first_ent():
    """ Updating in memory first entity """
    st.session_state.ent_1_boolean = True

def on_change_refresh_second_ent():
    """ Updating in memory first entity """
    st.session_state.ent_2_boolean = True

def on_click_refresh_ent():
    """ Updating in memory first entity """
    st.session_state.ent_1_boolean = False
    st.session_state.ent_2_boolean = False