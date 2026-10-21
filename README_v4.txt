v4: PV + wind + SMR + grid, 25-year horizon, hydrogen chain off by default.
Files: vpp_backend_v4.py, app_v4.py, test_v4.py, requirements.txt (keep in one folder).
Run:  pip install -r requirements.txt ; streamlit run app_v4.py
Upload solar (timestamp, ghi), wind (timestamp, wind_speed), load (timestamp, load_kw); press Run NSGA-II.
