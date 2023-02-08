from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None
# # a = 0
# # b = 0
# # c = 10
# # d = 10
# # tifile = Image.open("./image.tif")
# filename = "insert nac here"
# # tifdim = tifile.size
# tifdim = (6500, 162200)
# tifgeo = [(285.491278, 29.456955), (286.573891, 2.434635)] #a -> longitude, b -> latitude
# imgdim = (5064, 52224)
# imggeo = [(286.1, 10.7), (286.17,9.75)] 
# imggeo = [(285.5700,8.7900),(285.840,6.6700)]
# 
# scalew = tifdim[0]/(tifgeo[1][0] - tifgeo[0][0])
# scaleh = tifdim[1]/(tifgeo[1][1] - tifgeo[0][1])
# 
# scale2w = imgdim[0]/(imggeo[1][0] - imggeo[0][0])*0.42
# scale2h = imgdim[1]/(imggeo[1][1] - imggeo[0][1])*0.55
# 
# endco = ((imggeo[0][0] - tifgeo[0][0])*scalew,
# 		(imggeo[0][1] - tifgeo[0][1])*scaleh,
# 		(imggeo[1][0] - tifgeo[0][0])*scalew,
# 		(imggeo[1][1] - tifgeo[0][1])*scaleh)
# print(scalew)
# print(scaleh)
# print(scale2w)
# print(scale2h)
# print(endco)
# print(endco[2] - endco[0])
# print(endco[3] - endco[1])
# 
# Image.open("preview.png").crop(tuple([int(x/10) for x in endco])).show()
# 
# # tocrop = tifile.crop(enco)
# # tocrop.save("test.png")
# # tifile = Image.open("./image.tif")
# # pixscale = int((1000/nacimg.size[0])*tocrop.size[0])
# # 
# # nacimg = Image.open(f"./{filename}.IMG.png")
# # 
filename = "final2"

nacimg = Image.open("nac2.png")
tmcimg = Image.open("tmc.png")
ns = 96 #Nac Size
ts = int(ns*tmcimg.width/nacimg.width) #Tmc Size

print(ts)
path = os.path.join("images", filename)
pathh = os.path.join(path, "high")
pathl = os.path.join(path, "low")
# os.removedirs(pathh)
# os.removedirs(pathl)
os.makedirs(pathh)
os.makedirs(pathl)
print(nacimg.size)
k = 0
bad = []
for i in range(0, nacimg.width, ns):
	for j in range(0, nacimg.height, ns):
		print(i, j, k)
		print((i,j,i+ns, j+ns))
		print((int(i*ts/ns), int(j*ts/ns), int(i*ts/ns) + ts, int(j*ts/ns) + ts))
		cropped = tmcimg.crop((int(i*ts/ns), int(j*ts/ns), int(i*ts/ns) + ts, int(j*ts/ns) + ts)).resize((24,24))
		if (cropped.histogram()[0] > 144):
			print(f"bad image for image {k}")
			bad.append(k)
			continue
		nacimg.crop((i,j,i+ns, j+ns)).save(f"{pathh}/image_{k}.png")
		cropped.save(f"{pathl}/image_{k}.png")
		k+=1
		
		
print(bad)



