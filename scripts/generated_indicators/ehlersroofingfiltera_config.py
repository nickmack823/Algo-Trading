{
  "name": "EhlersRoofingFilterA",
  "function": EhlersRoofingFilterA,
  "signal_function": None,
  "raw_function": EhlersRoofingFilterA,
  "description": "John Ehlers' roofing filter combining a high-pass and low-pass to isolate tradable trends; returns rfilt, trigger, hp, up, down.",
  "parameters": {
    "hp_length": 80,
    "lp_length": 40,
    "arrow_distance": 100.0,
    "point": 1.0
  },
  "parameter_space": {
    "hp_length": [40, 60, 80, 100, 120],
    "lp_length": [20, 30, 40, 50, 60]
  },
  "role": "Confirmation"
}