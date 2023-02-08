from PIL import Image

#this was used to view NAC image metadata (the first 5064 bytes) so that NAC images could be opened in Fiji (ImageJ) for viewing

with open("M106441626LC.img", mode="rb") as file:
	# while True:
# 		print(file.read(1024))
# 		input()
	bytez = file.read(5200)
	with open("test", mode="wb") as dump:
		dump.write(bytez)