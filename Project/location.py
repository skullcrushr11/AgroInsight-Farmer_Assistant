# import geocoder

# def get_current_location():
#     current_location = geocoder.ip("me")
#     if current_location.latlng:
#         latitude, longitude = current_location.latlng
#         address = current_location.address
#         return latitude, longitude, address
#     else:
#         return None

# def get_coordinates_from_address(address):
#     location = geocoder.osm(address)
#     if location.latlng:
#         return location.latlng
#     else:
#         return None

# def main():
#     print("Select an option:")
#     print("1. Check current location")
#     print("2. Input address")

#     choice = input("Enter your choice: ")

#     if choice == "1":
#         current_location = get_current_location()
#         if current_location:
#             latitude, longitude, address = current_location
#             print("Current Location:")
#             print("Latitude:", latitude)
#             print("Longitude:", longitude)
#             print("Address:", address)
#         else:
#             print("Failed to retrieve current location")

#     elif choice == "2":
#         input_address = input("Enter the address: ")
#         coordinates = get_coordinates_from_address(input_address)
#         if coordinates:
#             latitude, longitude = coordinates
#             print("Coordinates for the address:")
#             print("Latitude:", latitude)
#             print("Longitude:", longitude)
#         else:
#             print("Failed to retrieve coordinates for the address")

#     else:
#         print("Invalid choice")

# if __name__ == "__main__":
#     main()

import requests

API_KEY = "pk.403c5c772d05ee62b95a0e8d0a9a0edc"  # Replace with your API key

def get_location():
    url = f"https://opencellid.org/ajax/getPosition?key={API_KEY}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "lat" in data and "lon" in data:
                return data["lat"], data["lon"]
            else:
                print("❌ No location data found.")
                return None
        else:
            print(f"❌ API request failed. Status Code: {response.status_code}")
            print("Raw Response:", response.text)
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

# Get and print location
location = get_location()
if location:
    print(f"✅ Latitude: {location[0]}, Longitude: {location[1]}")
else:
    print("❌ Location could not be determined.")

