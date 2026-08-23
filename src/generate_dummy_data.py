"""
Synthetic / Dummy CCHAIN Dataset Generator
Generates a 100% schema-compliant synthetic test city for validating and testing
the Climate-Driven Vector-Borne Outbreak Surveillance pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd

def get_project_root() -> Path:
    curr = Path.cwd()
    if (curr / "data").exists():
        return curr
    file_parent = Path(__file__).resolve().parent.parent
    if (file_parent / "data").exists():
        return file_parent
    return curr

def generate_dummy_cchain_data(
    output_dir: Path = None,
    city_code: str = "PH990001000",
    city_name: str = "Synthetic Test City",
    num_rows_grid: int = 3,
    num_cols_grid: int = 4,
    start_date: str = "2003-01-01",
    end_date: str = "2022-12-31",
    random_seed: int = 42
):
    np.random.seed(random_seed)
    if output_dir is None:
        output_dir = get_project_root() / "data" / "dummy_test_city"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_brgys = num_rows_grid * num_cols_grid
    print(f"[*] Generating synthetic CCHAIN test dataset in: {output_dir}")
    print(f"[*] City: {city_name} ({city_code}) | {num_brgys} Barangays ({num_rows_grid}x{num_cols_grid} spatial grid)")
    print(f"[*] Date Range: {start_date} to {end_date}")

    # 1. BRGY METADATA & LOCATION.CSV
    brgys = []
    base_lon, base_lat = 124.60, 8.45
    dx, dy = 0.02, 0.02
    
    idx = 0
    for r in range(num_rows_grid):
        for c in range(num_cols_grid):
            idx += 1
            pcode = f"{city_code[:6]}{idx:03d}"
            name = f"Barangay Test-{idx:02d}"
            
            x0 = base_lon + c * dx
            x1 = x0 + dx
            y0 = base_lat + r * dy
            y1 = y0 + dy
            
            wkt_poly = f"POLYGON (({x0:.4f} {y0:.4f}, {x1:.4f} {y0:.4f}, {x1:.4f} {y1:.4f}, {x0:.4f} {y1:.4f}, {x0:.4f} {y0:.4f}))"
            area_sqkm = round(np.random.uniform(0.5, 4.5), 4)
            is_coastal = bool(r == 0)
            
            brgys.append({
                "adm1_en": "Region Synthetic",
                "adm1_pcode": "PH990000000",
                "adm2_en": "Synthetic Province",
                "adm2_pcode": "PH990000000",
                "adm3_en": city_name,
                "adm3_pcode": city_code,
                "adm4_en": name,
                "adm4_pcode": pcode,
                "brgy_total_area": area_sqkm,
                "brgy_is_coastal": is_coastal,
                "geometry": wkt_poly,
                "grid_r": r,
                "grid_c": c
            })
            
    df_brgy_meta = pd.DataFrame(brgys)
    loc_cols = ["adm1_en", "adm1_pcode", "adm2_en", "adm2_pcode", "adm3_en", "adm3_pcode", "adm4_en", "adm4_pcode", "brgy_total_area"]
    df_loc = df_brgy_meta[loc_cols].copy()
    df_loc.to_csv(output_dir / "location.csv", index=False)
    print(f"[+] Created location.csv ({len(df_loc)} rows)")

    # 2. BRGY_GEOGRAPHY.CSV
    geo_rows = []
    for i, b in df_brgy_meta.iterrows():
        geo_rows.append({
            "uuid": f"BGEOG{i:06d}",
            "adm4_pcode": b["adm4_pcode"],
            "date": "2003-01-01",
            "freq": "S",
            "brgy_total_area": b["brgy_total_area"],
            "brgy_distance_to_coast": 0.0 if b["brgy_is_coastal"] else round(float(b["grid_r"] * 2500.0), 2),
            "brgy_is_coastal": b["brgy_is_coastal"],
            "geometry": b["geometry"]
        })
    df_geo = pd.DataFrame(geo_rows)
    df_geo.to_csv(output_dir / "brgy_geography.csv", index=False)
    print(f"[+] Created brgy_geography.csv ({len(df_geo)} rows)")

    # 3. BUILDINGS & WORLDPOP POPULATION
    bldg_rows = []
    pop_rows = []
    
    for i, b in df_brgy_meta.iterrows():
        area = b["brgy_total_area"]
        built_up_pct = round(np.random.uniform(5.0, 75.0), 2)
        bldg_count = int(area * 1000 * (built_up_pct / 100.0) * np.random.uniform(0.8, 1.2))
        bldg_density = round(bldg_count / (area * 1e6), 8)
        bldg_mean_area = round(np.random.uniform(40.0, 120.0), 2)
        
        bldg_rows.append({
            "uuid": f"GOBLG{i:06d}",
            "adm4_pcode": b["adm4_pcode"],
            "date": "2023-01-01",
            "freq": "Y",
            "google_bldgs_count": float(bldg_count),
            "google_bldgs_area_total": round(bldg_count * bldg_mean_area, 2),
            "google_bldgs_area_mean": bldg_mean_area,
            "google_bldgs_count_lt100_sqm": float(int(bldg_count * 0.7)),
            "google_bldgs_count_100_200_sqm": float(int(bldg_count * 0.2)),
            "google_bldgs_count_gt_200_sqm": float(int(bldg_count * 0.1)),
            "google_bldgs_density": bldg_density,
            "google_bldgs_pct_built_up_area": built_up_pct
        })
        
        base_pop = int(area * np.random.uniform(2000, 15000))
        for yr in range(2000, 2023):
            curr_pop = int(base_pop * ((1.015) ** (yr - 2000)) * np.random.uniform(0.98, 1.02))
            pop_rows.append({
                "uuid": f"WDPOP{len(pop_rows):06d}",
                "adm4_pcode": b["adm4_pcode"],
                "date": f"{yr}-01-01",
                "freq": "Y",
                "pop_count_total": float(curr_pop),
                "pop_count_mean": float(curr_pop / 10),
                "pop_count_median": float(curr_pop / 10),
                "pop_count_stdev": float(curr_pop / 50),
                "pop_count_min": float(curr_pop / 100),
                "pop_count_max": float(curr_pop / 5),
                "pop_density_mean": round(float(curr_pop / area), 4),
                "pop_density_median": round(float(curr_pop / area), 4),
                "pop_density_stdev": 10.0,
                "pop_density_min": 100.0,
                "pop_density_max": 25000.0
            })

    pd.DataFrame(bldg_rows).to_csv(output_dir / "google_open_buildings.csv", index=False)
    pd.DataFrame(pop_rows).to_csv(output_dir / "worldpop_population.csv", index=False)
    print(f"[+] Created google_open_buildings.csv ({len(bldg_rows)} rows)")
    print(f"[+] Created worldpop_population.csv ({len(pop_rows)} rows)")

    # 4. DAILY CLIMATE ATMOSPHERE (ERA5-LAND)
    dates_daily = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(dates_daily)
    day_of_year = dates_daily.dayofyear.values
    
    seasonal_temp = 27.0 + 2.5 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
    seasonal_rain = 5.0 + 6.0 * np.maximum(0, np.sin(2 * np.pi * (day_of_year - 150) / 365.25))

    clim_records = []
    uuid_counter = 0

    print("[*] Generating daily climate simulation...")
    for b in df_brgy_meta.itertuples():
        pcode = b.adm4_pcode
        urban_boost = (b.brgy_is_coastal * -0.3) + 0.5 * (b.grid_r + b.grid_c) / (num_rows_grid + num_cols_grid)
        
        tmean = seasonal_temp + urban_boost + np.random.normal(0, 0.8, size=total_days)
        tmin = tmean - np.random.uniform(2.5, 4.5, size=total_days)
        tmax = tmean + np.random.uniform(3.0, 5.5, size=total_days)
        
        rh = np.clip(75.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 180) / 365.25) + np.random.normal(0, 5.0, size=total_days), 50.0, 98.0)
        heat_idx = tmean + (0.5555 * (6.11 * np.exp(5417.7530 * (1/273.16 - 1/(273.15 + tmean))) * (rh/100) - 10))
        
        rain_prob = np.clip(0.3 + 0.3 * (seasonal_rain / 11.0), 0.1, 0.8)
        is_rain_day = np.random.binomial(1, rain_prob, size=total_days)
        pr = np.round(is_rain_day * np.random.exponential(scale=seasonal_rain + 2.0, size=total_days), 2)
        
        wind_speed = np.round(np.random.uniform(1.2, 4.8, size=total_days), 2)
        solar_rad = np.round(np.random.uniform(140.0, 240.0, size=total_days), 2)
        uv_rad = np.round(solar_rad * 0.12, 2)

        for d_idx, dt in enumerate(dates_daily):
            uuid_counter += 1
            clim_records.append({
                "uuid": f"CATMS{uuid_counter:07d}",
                "adm4_pcode": pcode,
                "date": dt.strftime("%Y-%m-%d"),
                "freq": "D",
                "tave": round(float(tmean[d_idx]), 2),
                "tmin": round(float(tmin[d_idx]), 2),
                "tmax": round(float(tmax[d_idx]), 2),
                "heat_index": round(float(heat_idx[d_idx]), 2),
                "pr": float(pr[d_idx]),
                "wind_speed": float(wind_speed[d_idx]),
                "rh": round(float(rh[d_idx]), 2),
                "solar_rad": float(solar_rad[d_idx]),
                "uv_rad": float(uv_rad[d_idx])
            })

    df_clim = pd.DataFrame(clim_records)
    df_clim.to_csv(output_dir / "climate_atmosphere.csv", index=False)
    print(f"[+] Created climate_atmosphere.csv ({len(df_clim)} daily rows)")

    # 5. HEALTH RECORDS (DENGUE)
    df_clim["month_dt"] = pd.to_datetime(df_clim["date"]).dt.to_period("M").dt.to_timestamp()
    monthly_clim = df_clim.groupby(["adm4_pcode", "month_dt"], as_index=False).agg(
        pr_sum=("pr", "sum"),
        tave_mean=("tave", "mean"),
        hi_mean=("heat_index", "mean")
    ).sort_values(by=["adm4_pcode", "month_dt"]).reset_index(drop=True)

    monthly_clim["pr_lag2m"] = monthly_clim.groupby("adm4_pcode")["pr_sum"].shift(2)
    monthly_clim["hi_lag2m"] = monthly_clim.groupby("adm4_pcode")["hi_mean"].shift(2)
    
    bldg_pct_map = dict(zip(df_brgy_meta["adm4_pcode"], [b["google_bldgs_pct_built_up_area"] for b in bldg_rows]))
    pop_map = dict(zip(df_brgy_meta["adm4_pcode"], [b["pop_count_total"] for b in pop_rows if "2020" in b["date"]]))
    
    health_records = []
    h_uuid = 0
    
    print("[*] Simulating biological lag-driven dengue epidemics...")
    for row in monthly_clim.itertuples():
        pcode = row.adm4_pcode
        dt_str = row.month_dt.strftime("%Y-%m-%d")
        
        pr_lag = row.pr_lag2m if pd.notna(row.pr_lag2m) else 100.0
        hi_lag = row.hi_lag2m if pd.notna(row.hi_lag2m) else 30.0
        bldg_pct = bldg_pct_map.get(pcode, 30.0)
        pop = pop_map.get(pcode, 15000)
        
        climate_force = ((pr_lag / 150.0) ** 1.3) * ((hi_lag / 28.0) ** 2.0)
        urban_multiplier = (bldg_pct / 40.0) * (pop / 20000.0)
        
        expected_lambda = np.clip(1.2 * climate_force * urban_multiplier, 0.2, 45.0)
        
        yr = row.month_dt.year
        if yr in [2006, 2012, 2016, 2019, 2022]:
            expected_lambda *= 2.2
            
        cases = int(np.random.poisson(lam=expected_lambda))
        deaths = int(np.random.binomial(cases, p=0.008)) if cases > 0 else 0
        
        h_uuid += 1
        health_records.append({
            "uuid": f"DLGUT{h_uuid:06d}",
            "freq": "M",
            "date": dt_str,
            "source_name": "SYNTHETIC-HEALTH-SYS",
            "source_filename": f"Synthetic_{yr}_Morbidity",
            "adm3_pcode": city_code,
            "adm4_pcode": pcode,
            "disease_icd10_code": "A90",
            "disease_common_name": "DENGUE FEVER",
            "sex": "TOTAL",
            "age_group": "ALL",
            "case_total": cases,
            "death_total": deaths
        })

    df_health = pd.DataFrame(health_records)
    df_health.to_csv(output_dir / "disease_lgu_disaggregated_totals.csv", index=False)
    print(f"[+] Created disease_lgu_disaggregated_totals.csv ({len(df_health)} monthly records)")
    print(f"[SUCCESS] Synthetic test dataset ready in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    generate_dummy_cchain_data()
