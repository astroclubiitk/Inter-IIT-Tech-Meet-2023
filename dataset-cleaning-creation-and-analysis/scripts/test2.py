from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = 4126498656
# filename = "M106488417LE"
# filename = "M106441626LC"
# width, height = 2532, 30720
width, height = 5064, 40960

# with open(f"{filename}.IMG", mode="rb") as file:
# 	test = Image.frombytes("L", (width, height), file.read())
# 	for i in range(1, int(height/384)+ 1):
# 		for j in range(1, int(width/384) + 1):
# 			print(f"{i} {j}")
# 			cropped = test.crop((384*(j-1), 384*(i-1), 384*j, 384*i))
# 			cropped.save(f"images/{filename}/high/image{i}_{j}.png")
# 			cropped.resize((64, 64)).save(f"images/{filename}/low/image{i}_{j}.png")


big_size = 1024
small_size = 256
pic_names = ["M106441626LE","M1142603254LE","M1142603254RE","M1221560173RE","M1221615756LE","M1221616032LE"]

for filename in pic_names:
	path = os.path.join("images_hotfix", filename)
	pathh = os.path.join(path, "high")
	pathl = os.path.join(path, "low")
	os.makedirs(pathh)
	os.makedirs(pathl)
	k = 0
	test = Image.open(f"{filename}.IMG.png")
	for i in range(1, int(height/big_size)+ 1):
		for j in range(1, int(width/big_size) + 1):
			if (j == 1):
				continue
			print(f"{i} {j}")
			cropped = test.crop((big_size*(j-1), big_size*(i-1), big_size*j, big_size*i))
			cropped.save(f"{pathh}/image_{k}.png")
			cropped.resize((small_size, small_size)).save(f"{pathl}/image_{k}.png")
			k += 1