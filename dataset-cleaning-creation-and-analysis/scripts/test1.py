from PIL import Image

Image.MAX_IMAGE_PIXELS = 4126498656

width, height = 2532, 30720
width, height = 5064, 40960



# with open("M106441626LC.IMG", mode="rb") as file:
# 	test = Image.frombytes("F", (width, height), file.read())
# 	test.show()
	# for i in range(1, int(height/2048)+ 1):
# 		print(i)
# 		cropped = test.crop((0, 2048*(i-1), 2048, 2048*i))
# 		cropped.save(f"images/high/image{i}.png")
# 		cropped.resize((256, 256)).save(f"images/low/image{i}.png")
	
	
# test = Image.open("ch2_tmc_ndn_20210514T0803114100_d_oth_d18/data/derived/20210514/ch2_tmc_ndn_20210514T0803114100_d_oth_d18.tif")
# test.show()

test = Image.open("M106441626LC.png")
test.save("test.tif")