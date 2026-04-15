import pandas as pd
import numpy as np
import datetime

def generate_dataset(num_records=100000):
    np.random.seed(42)

    # 1. Coordinate clustering (5 main ocean gyres + some random noise)
    # North Pacific, South Pacific, North Atlantic, South Atlantic, Indian Ocean
    gyres_centers = [
        (35, -140),  # North Pacific
        (-30, -120), # South Pacific
        (30, -40),   # North Atlantic
        (-30, -20),  # South Atlantic
        (-25, 80)    # Indian Ocean
    ]
    
    # 80% data around gyres (hotspots), 20% random spread
    gyre_samples = int(num_records * 0.8)
    random_samples = num_records - gyre_samples
    
    gyre_indices = np.random.randint(0, len(gyres_centers), gyre_samples)
    lats = []
    lons = []
    
    for idx in gyre_indices:
        lat, lon = gyres_centers[idx]
        lats.append(np.random.normal(loc=lat, scale=10))      # Spread around gyre
        lons.append(np.random.normal(loc=lon, scale=15))

    # Random ocean noise
    lats.extend(np.random.uniform(-60, 60, random_samples))
    lons.extend(np.random.uniform(-180, 180, random_samples))

    # Features according to NOAA & Copernicus:
    # "sea surface temperature, salinity, wind speed, and ocean current data"
    # "ocean current velocity, chlorophyll concentration, and wave height"
    # "Microplastic concentration"
    
    sea_surface_temp = np.random.normal(loc=18.0, scale=8.0, size=num_records) # Celsius
    salinity = np.random.normal(loc=35.0, scale=1.5, size=num_records)  # PSU
    wind_speed = np.random.gamma(shape=2.0, scale=3.0, size=num_records) # m/s
    ocean_current_velocity = np.random.gamma(shape=1.5, scale=0.3, size=num_records) # m/s
    chlorophyll_concentration = np.random.exponential(scale=0.5, size=num_records) # mg/m3
    wave_height = np.random.gamma(shape=2.0, scale=1.0, size=num_records) # m
    
    # Missing temporal and spatial features
    start_date = datetime.date(2023, 1, 1)
    dates = [start_date + datetime.timedelta(days=int(np.random.randint(0, 365))) for _ in range(num_records)]
    distance_from_coastline = np.random.exponential(scale=100.0, size=num_records) # km
    distance_from_river_mouth = np.random.exponential(scale=300.0, size=num_records) # km
    
    # Compute a realistic microplastic concentration logic
    # Higher concentration near gyre centers, influenced heavily by ocean current and inversely by wind speed
    base_concentration = np.random.gamma(shape=1.0, scale=5.0, size=num_records)
    
    # Add strong signal if in Gyre
    is_gyre = np.array([1 if i < gyre_samples else 0 for i in range(num_records)])
    microplastic_concentration = base_concentration + (is_gyre * 25.0) \
                                + (ocean_current_velocity * 10) \
                                - (wind_speed * 0.5)
    
    # Ensure no negative concentration
    microplastic_concentration = np.clip(microplastic_concentration, 0, None)
    
    df = pd.DataFrame({
        "Date": dates,
        "Latitude": lats,
        "Longitude": lons,
        "Sea_Surface_Temperature": sea_surface_temp,
        "Salinity": salinity,
        "Wind_Speed": wind_speed,
        "Ocean_Current_Velocity": ocean_current_velocity,
        "Chlorophyll_Concentration": chlorophyll_concentration,
        "Wave_Height": wave_height,
        "Distance_from_Coastline": distance_from_coastline,
        "Distance_from_River_Mouth": distance_from_river_mouth,
        "Microplastic_Concentration": microplastic_concentration
    })
    
    df.to_csv("ocean_sensor_dataset.csv", index=False)
    print(f"Successfully generated 'ocean_sensor_dataset.csv' with {num_records} rows.")
    print("Features included:", df.columns.tolist())

if __name__ == "__main__":
    generate_dataset()
