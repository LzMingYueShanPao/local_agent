from common import constants
import requests


url = "https://api.qweather.com/v7/weather/3d?"
params = {
    'location': '101010100',
    'key': constants.WEATHER_KEY,
}
response = requests.get(url, params=params)
print('response=',response)
