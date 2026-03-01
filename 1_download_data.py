from roboflow import Roboflow
rf = Roboflow(api_key="tw2bwUpDJOrZnWpKFL3X") # Replace with your own API_KEY when you create a roboflow account
project = rf.workspace("rcr-mjqgv").project("rock-paper-scissors-sxsw-jdhtk")
version = project.version(1)
dataset = version.download("yolo26")