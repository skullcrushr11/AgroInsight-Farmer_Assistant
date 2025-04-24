import requests
from datetime import datetime, timedelta
import sys
from collections import Counter

def get_location():
    """Get latitude/longitude from IP using ip-api.com."""
    try:
        response = requests.get("http://ip-api.com/json")
        data = response.json()
        if data["status"] == "success":
            return data["lat"], data["lon"], data["city"], data["country"]
        else:
            print("Error: Failed to get location from ip-api.com")
            return None, None, None, None
    except Exception as e:
        print(f"Error getting location: {str(e)}")
        return None, None, None, None

def classify_condition(precip, cloud_cover, sunshine_hours):
    """Classify daily weather as rainy, cloudy, sunny, or other."""
    if precip > 0.5:
        return "Rainy"
    elif cloud_cover > 60:
        return "Cloudy"
    elif cloud_cover < 30 and sunshine_hours > 4:
        return "Sunny"
    else:
        return "Mixed"

def get_weather_data(lat, lon, start_date, end_date):
    """Fetch weather data from Open-Meteo API for a given date range."""
    if not lat or not lon:
        print("Error: Invalid latitude/longitude")
        return None
    
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
            "cloudcover_mean,sunshine_duration&timezone=auto"
        )
        print(f"Fetching data for {start_date} to {end_date}...")
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            print(f"API Error: {data.get('reason', 'Unknown error')}")
            return None
        if "daily" not in data or not data["daily"].get("time"):
            print("Error: No valid weather data returned from Open-Meteo")
            return None
        
        return data["daily"]
    
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

def process_weather_data(daily):
    """Process daily weather data into required metrics."""
    if not daily:
        return None
    
    monthly_rainfall = {}
    monthly_conditions = {}
    valid_temps = []
    valid_humidity = []
    
    for date, temp, humidity, precip, cloud, sunshine in zip(
        daily["time"],
        daily["temperature_2m_mean"],
        daily["relative_humidity_2m_mean"],
        daily["precipitation_sum"],
        daily["cloudcover_mean"],
        daily["sunshine_duration"]
    ):
        if any(x is None for x in [temp, humidity, precip, cloud, sunshine]):
            continue
        month = date[:7]
        monthly_rainfall[month] = monthly_rainfall.get(month, 0) + precip
        sunshine_hours = sunshine / 3600
        condition = classify_condition(precip, cloud, sunshine_hours)
        if month not in monthly_conditions:
            monthly_conditions[month] = []
        monthly_conditions[month].append(condition)
        valid_temps.append(temp)
        valid_humidity.append(humidity)
    
    if not valid_temps:
        return None
    
    avg_temp = sum(valid_temps) / len(valid_temps)
    avg_humidity = sum(valid_humidity) / len(valid_humidity)
    total_rainfall = sum(monthly_rainfall.values())
    highest_rainfall = max(monthly_rainfall.values()) if monthly_rainfall else None
    
    all_conditions = []
    for conditions in monthly_conditions.values():
        all_conditions.extend(conditions)
    condition_counter = Counter(all_conditions)
    dominant_condition = max(condition_counter, key=condition_counter.get) if condition_counter else None
    
    return {
        "monthly_rainfall_mm": {k: round(v, 2) for k, v in monthly_rainfall.items()},
        "highest_monthly_rainfall_mm": round(highest_rainfall, 2) if highest_rainfall else None,
        "avg_temperature_celsius": round(avg_temp, 2),
        "avg_relative_humidity_percent": round(avg_humidity, 2),
        "moisture": round(avg_humidity / 100, 4),
        "total_rainfall_mm": round(total_rainfall, 2),
        "dominant_condition": dominant_condition
    }

def get_date_ranges():
    """Calculate date ranges based on current month."""
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Previous year, next 3 months (e.g., Apr 2025 -> Apr, May, Jun 2024)
    prev_year = current_year - 1
    start_month_next = now.replace(month=current_month, year=prev_year, day=1)
    end_month_next = start_month_next + timedelta(days=89)
    next_3_start = start_month_next.strftime("%Y-%m-%d")
    next_3_end = end_month_next.strftime("%Y-%m-%d")
    
    # Previous year, last 3 months (e.g., Apr 2025 -> Feb, Mar, Apr 2024)
    start_month_last = now.replace(month=current_month, year=prev_year, day=1) - timedelta(days=60)
    end_month_last = now.replace(month=current_month, year=prev_year, day=1) + timedelta(days=29)
    last_3_start = start_month_last.strftime("%Y-%m-%d")
    last_3_end = end_month_last.strftime("%Y-%m-%d")
    
    # Present year, last 4 months (e.g., Apr 2025 -> Dec 2024, Jan, Feb, Mar 2025)
    start_month_current = now.replace(day=1) - timedelta(days=120)
    end_month_current = now - timedelta(days=1) # Limit to today to avoid future data
    current_4_start = start_month_current.strftime("%Y-%m-%d")
    current_4_end = end_month_current.strftime("%Y-%m-%d")
    
    # Fallback: Previous year equivalent for current period
    start_month_fallback = start_month_current.replace(year=prev_year)
    end_month_fallback = end_month_current.replace(year=prev_year)
    fallback_4_start = start_month_fallback.strftime("%Y-%m-%d")
    fallback_4_end = end_month_fallback.strftime("%Y-%m-%d")
    
    return {
        "prev_next_3": (next_3_start, next_3_end),
        "prev_last_3": (last_3_start, last_3_end),
        "current_last_4": (current_4_start, current_4_end),
        "fallback_last_4": (fallback_4_start, fallback_4_end)
    }

def main():
    print("Fetching weather data for your project metrics...")
    
    # Get location
    lat, lon, city, country = get_location()
    if not lat or not lon:
        print("Cannot fetch weather data without a valid location.")
        sys.exit(1)
    
    print(f"\nLocation: {city}, {country} (Latitude: {lat}, Longitude: {lon})")
    
    # Get date ranges
    date_ranges = get_date_ranges()
    
    # Fetch and process data for each period
    periods = {
        "prev_next_3": "Previous Year Next 3 Months (e.g., Apr-May-Jun 2024)",
        "prev_last_3": "Previous Year Last 3 Months (e.g., Feb-Mar-Apr 2024)",
        "current_last_4": "Present Year Last 4 Months (e.g., Dec 2024-Mar 2025)",
        "fallback_last_4": "Fallback: Previous Year Same Period (e.g., Dec 2023-Mar 2024)"
    }
    
    weather_results = {}
    for period, (start_date, end_date) in date_ranges.items():
        daily = get_weather_data(lat, lon, start_date, end_date)
        weather_results[period] = process_weather_data(daily)
    
    # Display results
    print("\n=== Crop Recommendation Metrics ===")
    prev_next = weather_results.get("prev_next_3")
    if prev_next:
        print(f"Period: {periods['prev_next_3']}")
        print(f"Highest Monthly Rainfall: {prev_next['highest_monthly_rainfall_mm']} mm")
        print(f"Average Temperature: {prev_next['avg_temperature_celsius']}°C")
        print(f"Average Relative Humidity: {prev_next['avg_relative_humidity_percent']}%")
    else:
        print("No data available for previous year next 3 months.")
    
    print("\n=== Fertilizer 1 Metrics ===")
    if prev_next:
        print(f"Period: {periods['prev_next_3']}")
        print(f"Average Temperature: {prev_next['avg_temperature_celsius']}°C")
    else:
        print("No data available for previous year next 3 months.")
    
    print("\n=== Fertilizer 2 Metrics ===")
    if prev_next:
        print(f"Period: {periods['prev_next_3']}")
        print(f"Average Temperature: {prev_next['avg_temperature_celsius']}°C")
        print(f"Highest Monthly Rainfall: {prev_next['highest_monthly_rainfall_mm']} mm")
        print(f"Soil Moisture (Humidity/100): {prev_next['moisture']}")
    else:
        print("No data available for previous year next 3 months.")
    
    print("\n=== Yield Prediction Metrics ===")
    current_last = weather_results.get("current_last_4")
    if current_last:
        print(f"Period: {periods['current_last_4']}")
        print(f"Total Rainfall: {current_last['total_rainfall_mm']} mm")
        print(f"Average Temperature: {current_last['avg_temperature_celsius']}°C")
        print(f"General Weather Condition: {current_last['dominant_condition']}")
    else:
        print("Warning: No data available for present year last 4 months. Using fallback.")
        fallback_last = weather_results.get("fallback_last_4")
        if fallback_last:
            print(f"Period: {periods['fallback_last_4']}")
            print(f"Total Rainfall: {fallback_last['total_rainfall_mm']} mm")
            print(f"Average Temperature: {fallback_last['avg_temperature_celsius']}°C")
            print(f"General Weather Condition: {fallback_last['dominant_condition']}")
        else:
            print("No fallback data available either.")

if __name__ == "__main__":
    main()