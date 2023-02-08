from PIL import Image

with open("M106441626LC.img", mode="rb") as file:
	# while True:
# 		print(file.read(1024))
# 		input()
	bytez = file.read(5200)
	with open("test", mode="wb") as dump:
		dump.write(bytez)