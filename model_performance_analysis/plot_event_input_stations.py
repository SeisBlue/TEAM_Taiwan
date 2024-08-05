import pandas as pd
from analysis import TriggeredMap

# plot input station map
mask_after_sec = 10
eq_id = 25900
prediction_with_info = pd.read_csv(
    f"../predict/station_blind_noVs30_bias2closed_station_2016/{mask_after_sec}_sec_ensemble_510_with_all_info.csv"
)
record_prediction = prediction_with_info.query(f"EQ_ID=={eq_id}")
first_trigger_time = min(record_prediction["p_picks"])
input_station = record_prediction[
    record_prediction["p_picks"] < first_trigger_time + (mask_after_sec * 200)
]


if len(input_station) >= 25:
    input_station = input_station[:25]

fig, ax = TriggeredMap.plot_station_map(
    trace_info=input_station,
    sec=mask_after_sec,
    EQ_ID=eq_id,
    pad=100,
)

# fig.savefig(
#     f"../paper_image/eqid{eq_id}_{mask_after_sec}_sec_station_input.png",dpi=300
# )
