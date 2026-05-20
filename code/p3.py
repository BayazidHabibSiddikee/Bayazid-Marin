import os
os.system("start microsoft.windows.camera:")
import time
time.sleep(3)
os.system("taskkill /IM WindowsCamera.exe /F")
