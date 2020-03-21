import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_2.json', scope)
client = gspread.authorize(credentials)
sheet = client.open('DL_Exp1').sheet1
row = [1, 2, 3]
index = 1
sheet.insert_row(row, index)
