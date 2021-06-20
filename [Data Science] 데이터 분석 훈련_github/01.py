#웹크롤링 위한 라이브러리 모두 호출
import selenium
from selenium import webdriver
import os
import time
import random
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.keys import Keys

#엑셀에 파일 다운로드 받기 위한 라이브러리 호출
import openpyxl
from openpyxl import load_workbook
from openpyxl import Workbook
from PIL import Image
from io import BytesIO
import requests
